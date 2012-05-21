import numm
import numpy
import gst
import colorsys
import cv2
import os

class FrameEater(object):
    def peek(self):
        "get output without resetting state"
        pass
    def pop(self):
        "get output and reset state"
        out = self.peek()
        self.reset()
        return out
    def reset(self):
        "reset state"
        pass
    def process(self, frame):
        "take new frame"
        pass
    def serialize(self, fpath):
        "write to disk"
        pass

class ImageToImage(FrameEater):
    extension = "png"
    def serialize(self, fpath):
        numm.np2image(self.peek(), fpath + "." + self.extension)

class Slitscan(ImageToImage):
    def __init__(self):
        self.slits = []
    def process(self, frame):
        self.slits.append(frame[:,frame.shape[1]/2])
    def peek(self):
        return numpy.array(self.slits).transpose(1,0,2)
    def reset(self):
        self.slits = []

class Oxscan(Slitscan):
    def process(self, frame):
        self.slits.append(frame.mean(axis=1).astype(numpy.uint8))

class FirstFrame(ImageToImage):
    def __init__(self):
        self.first = None
    def process(self, frame):
        if self.first is None:
            self.first = frame
    def peek(self):
        return self.first
    def reset(self):
        self.first = None

class LastFrame(ImageToImage):
    def process(self, frame):
        self.last = frame
    def peek(self):
        return self.last

class Composite(ImageToImage):
    def __init__(self):
        self.comp = None
        self.nframes = 1

    def process(self, frame):
        if self.comp is None:
            self.comp = frame.astype(int)
        else:
            self.nframes += 1
            self.comp += frame

    def peek(self):
        return (self.comp / self.nframes).astype(numpy.uint8)
    def reset(self):
        self.comp = None
        self.nframes = 1

class ImageToMath(FrameEater):
    def serialize(self, fpath):
        numpy.savez(fpath + ".npz", **self.peek())

class Flow(ImageToMath):
    def __init__(self):
        self.points = None
        self.offset = 0
        self.surf = cv2.SURF()

    def process(self, frame):
        gray = frame.mean(axis=2).astype(numpy.uint8)

        if self.points is None:
            keypoints, descriptions = self.surf.detect(gray, None, False)

            if len(keypoints) < 2:
                self.offset += 1
                return

            self.points = numpy.array([list(X.pt) + [X.size, X.angle, X.response] for X in keypoints])

            self.features = descriptions.reshape((len(keypoints), -1))

            self.prevpts = numpy.array([X.pt for X in keypoints], dtype=numpy.float32).reshape((-1,1,2))
            self.flows = [self.prevpts.reshape((-1,2))]

        else:
            nextpts, status, err = cv2.calcOpticalFlowPyrLK(self.prevframe, gray, self.prevpts, None)
            valid_flows = status.reshape(-1) == 1
            new_flows = numpy.zeros(self.flows[0].shape)
            new_flows[valid_flows] = nextpts.reshape((-1,2))[valid_flows]
            self.flows.append(new_flows)
            self.prevpts = nextpts

        self.prevframe = gray

    def peek(self):
        if self.points is not None:
            return {"points": self.points,
                    "features": self.features,
                    "flows": self.flows,
                    "offset": self.offset}
        else: 
            return {}

    def reset(self):
        self.points = None
        self.offset = 0

class Histograms(ImageToMath):
    "hue, saturation, & value histograms of a random (!) frame"
    # XXX: diy with numpy ops
    # XXX: should be composite or mid-frame?

    def __init__(self):
        self.frame = None

    def process(self, frame):
        if self.frame is None or numpy.random.random() < 0.1:
            self.frame = frame
    def peek(self):
        out = {}
        frame = self.frame
        hsv = []
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                pixel = frame[y,x] / 255.0
                hsv.append(colorsys.rgb_to_hsv(*pixel.tolist()))

        # compute histogram
        hsv = numpy.array(hsv)
        for i, key in enumerate(('hue_hist', 'sat_hist', 'val_hist')):
            histogram, _boundaries = numpy.histogram(hsv[:,i], bins=10)
            histogram = histogram / float(len(hsv)) # normalize w/r/t number of pixels
            out[key] = histogram

        avg_pixel = frame.mean(axis=0).mean(axis=0) / 255.0
        out['avg_hsv'] = numpy.array([int(X*255) for X in colorsys.rgb_to_hsv(*avg_pixel.tolist())])
        return out

def discontinuity(a, b, threshold=0.2, hist_scale=0.5, col_threshold=10):
    # heuristic to determine if there is a discontinuity between a and b

    # compare b&w histograms
    a_hist, _boundaries = numpy.histogram(a.mean(axis=2), bins=10, range=(0, 255))
    b_hist, _boundaries = numpy.histogram(b.mean(axis=2), bins=10, range=(0, 255))

    hist_dist = hist_scale * sum(abs(a_hist - b_hist)) / float(a.shape[0]*a.shape[1])

    # compare averaged columns
    a_cols = a.mean(axis=1)
    b_cols = b.mean(axis=1)
    col_dist = sum((abs(a_cols - b_cols).mean(axis=1) > col_threshold).flat) / float(a.shape[1])

    return hist_dist + col_dist > threshold

class Analysis(FrameEater):
    def __init__(self, everyframe=None, post=None):
        # these receive frames, but aren't yielded or serialized.
        # you are responsible ...
        self.everyframe = everyframe
        # called on output every "peek"
        self.post = None

        self.machines = {"slitscan":Slitscan(), 
                         "oxscan": Oxscan(),
                         "first_frame": FirstFrame(),
                         "last_frame": LastFrame(),
                         "composite": Composite(), 
                         "histograms": Histograms(), 
                         "flow": Flow()}

    def process(self, frame):
        for m in self.machines.values():
            m.process(frame)
        if self.everyframe is not None:
            self.everyframe.process(frame)

    def peek(self):
        out = {k: v.peek() for k,v in self.machines.items()}
        if self.post is not None:
            return self.post(out)
        return out

    def reset(self):
        for m in self.machines.values():
            m.reset()

    def serialize(self, dirname):
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        for name, machine in self.machines.items():
            machine.serialize(os.path.join(dirname, name))

def analyze(path, height=96, min_n_frames=25, max_n_frames=3000):
    analysis = Analysis()
    nframes = 0
    cur_is_hard = True

    prevframe  = None
    prevprevframe = None
    for frame in numm.video_frames(path, height=height, fps=25):
        if nframes>min_n_frames and discontinuity(prevframe, frame) and discontinuity(prevprevframe, frame):
            # HARD CUT
            yield (analysis.pop(), cur_is_hard)
            nframes = 0
            cur_is_hard = True

        elif nframes >= max_n_frames:
            # SOFT CUT (max_n_frames)
            # XXX: bring back metric to segment on slow changes
            yield (analysis.pop(), cur_is_hard)
            cur_is_hard = False
            nframes = 0

        analysis.process(frame)
        nframes += 1
        prevprevframe = prevframe
        prevframe = frame
    yield (analysis.pop(), cur_is_hard)

class EveryNSecs(FrameEater):
    def __init__(self, outpattern="m5_%d.png", nsecs=5):
        self.outpattern = outpattern
        self.nsecs=nsecs
        self.p=0.2

    def process(self, frame):
        t = frame.timestamp / float(gst.SECOND)
        if t % self.nsecs < self.p % self.nsecs:
            numm.np2image(frame, self.outpattern % (t))
        self.p=t

def serialize(videopath, directory=None, min_n_frames=25, max_n_frames=3000):
    import json, time

    t0 = time.time()

    if directory is None:
        directory = videopath + '.analysis'
    absdir = os.path.join(directory, 'abs')

    if not os.path.isdir(directory):
        os.makedirs(directory)
    if not os.path.isdir(absdir):
        os.makedirs(absdir)

    doc = []
    def cut_doc(ishard, firstframe, nextframe):
        return {'start': firstframe.timestamp / float(gst.SECOND),
                'duration': (nextframe.timestamp - firstframe.timestamp) / float(gst.SECOND),
                'hard': ishard
}

    everyframe= EveryNSecs(outpattern=os.path.join(absdir, "mod_%d.png"), nsecs=5)

    # FIXME: repeated code (so as to serialize & post-process ...)
    analysis = Analysis(everyframe=everyframe)
    nframes = 0
    ncuts = 0
    cur_is_hard = True

    prevframe  = None
    firstframe = None
    for frame in numm.video_frames(videopath, height=96, fps=25):
        if firstframe is None:
            firstframe = frame

        if nframes>min_n_frames and discontinuity(prevframe, frame):
            # HARD CUT
            doc.append(cut_doc(cur_is_hard, firstframe, frame))
            analysis.serialize(os.path.join(directory, str(ncuts)))
            analysis.reset()
            firstframe = frame
            nframes = 0
            ncuts += 1
            cur_is_hard = True

        elif nframes >= max_n_frames:
            # SOFT CUT (max_n_frames)
            # XXX: bring back metric to segment on slow changes
            doc.append(cut_doc(cur_is_hard, firstframe, frame))
            analysis.serialize(os.path.join(directory, str(ncuts)))
            analysis.reset()
            firstframe = frame
            cur_is_hard = False
            nframes = 0
            ncuts += 1

        analysis.process(frame)
        nframes += 1
        prevframe = frame

    doc.append(cut_doc(cur_is_hard, firstframe, frame))
    analysis.serialize(os.path.join(directory, str(ncuts)))
    analysis.reset()

    json.dump(doc, open(os.path.join(directory, 'video.json'), 'w'))

    dur = doc[-1]['start'] + doc[-1]['duration']
    dt = time.time() - t0
    ratio = dur / dt

    print 'analyzed a %ds video in %ds (%.2fx)' % (dur, dt, ratio)


if __name__=='__main__':
    import sys
    serialize(sys.argv[1])

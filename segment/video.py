import numm
import numpy
import gst
import colorsys
import cv2

class Cut(object):
    def __init__(self):
        self.doc = {}
        self.images = {}
        self.absimages = {}
        self.arrays = {}

def discontinuity(a, b, threshold=0.1, col_threshold=10):
    # heuristic to determine if there is a discontinuity between a and b

    # compare b&w histograms
    a_hist, _boundaries = numpy.histogram(a.mean(axis=2), bins=10, range=(0, 255))
    b_hist, _boundaries = numpy.histogram(b.mean(axis=2), bins=10, range=(0, 255))

    hist_dist = sum(abs(a_hist - b_hist)) / float(a.shape[0]*a.shape[1])

    # compare averaged columns
    a_cols = a.mean(axis=1)
    b_cols = b.mean(axis=1)
    col_dist = sum((abs(a_cols - b_cols).mean(axis=1) > col_threshold).flat) / float(a.shape[1])

    return hist_dist + col_dist > threshold

def slitscans(np, d):
    d.images['left_slit'] = np[:,:,0].transpose(1,0,2)
    d.images['right_slit'] = np[:,:,-1].transpose(1,0,2)
    d.images['mid_slit'] = np[:,:,np.shape[2]/2].transpose(1,0,2)
    d.images['avg_slit'] = np.mean(axis=2).transpose(1,0,2).astype(numpy.uint8)

def hsv_metrics(np, d):
    frame = np[np.shape[0]/2]

    # transform into HSV
    # XXX: diy with numpy ops
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
        d.arrays[key] = histogram.tolist()

    avg_pixel = frame.mean(axis=0).mean(axis=0) / 255.0
    d.doc['avg_hsv'] = [int(X*255) for X in colorsys.rgb_to_hsv(*avg_pixel.tolist())]

def thumbstrips(np, d, num_strips=3):
    max_width = np.shape[2]
    step = max_width / num_strips
    periods = [int(step*X) for X in range(1, num_strips+1)]
    for p_idx, period in enumerate(periods):
        assert period <= max_width
        out = numpy.zeros((np.shape[1], np.shape[0], 3), numpy.uint8)
        for dx, fr in enumerate(np):
            if dx % period > 0:
                continue
            # center
            sx = max_width/2 - period/2
            slice_width = min(out.shape[1]-dx, period)

            out[:,dx:dx+slice_width] = fr[:,sx:sx+slice_width]
        d.images['thumbstrip-%d' % (p_idx)] = out

def frames(np, d):
    d.images['composite_frame'] = np.mean(axis=0).astype(numpy.uint8)
    d.images['first_frame'] = np[0]
    d.images['last_frame'] = np[-1]
    d.images['mid_frame'] = np[np.shape[0]/2]

def moduloframes(np, d, nsecs=5):
    # save thumbnails every five seconds, irrespective of segmentation

    prev_t = 0.02
    for fr in np:
        t = fr.timestamp / float(gst.SECOND)
        if t%nsecs < prev_t%nsecs:
            d.absimages["mod_%d" % (t)] = fr
        prev_t = t

def features(np, d):
    # compute features on the first frame, and track their motion through the cut

    grayscale = np.mean(axis=3).astype(numpy.uint8)
    firstframe = grayscale[0]

    surf = cv2.SURF(400)
    keypoints, descriptions = surf.detect(firstframe, None, False)
    d.arrays['keypoints'] = numpy.array([list(X.pt).extend([X.size, X.angle, X.response]) for X in keypoints])
    if len(keypoints) == 0:
        print 'no features'
        return
    d.arrays['keypoint_features'] = descriptions.reshape((len(keypoints), -1))

    prevframe = firstframe
    prevpts = numpy.array([X.pt for X in keypoints], dtype=numpy.float32).reshape((-1,1,2))
    flows = numpy.zeros((len(keypoints), np.shape[0], 2))
    flows[:,0] = prevpts.reshape((-1,2))

    for idx,frame in enumerate(grayscale[1:]):
        nextpts, status, err = cv2.calcOpticalFlowPyrLK(prevframe, frame, prevpts, None)
        valid_flows = status.reshape(-1) == 1
        flows[:,idx+1][valid_flows] = nextpts.reshape((-1,2))[valid_flows]
        prevframe = frame
        prevpts = nextpts
    d.arrays['flows'] = flows

def cut_from_frames(cur_cut, is_hard, end_time=None):
    d = Cut()

    d.doc['hardcut'] = is_hard;

    d.doc['start'] = cur_cut[0].timestamp / float(gst.SECOND)

    if end_time is None:
        d.doc['duration'] = (cur_cut[-1].timestamp - cur_cut[0].timestamp) / float(gst.SECOND)
    else:
        d.doc['duration'] = (end_time - cur_cut[0].timestamp) / float(gst.SECOND)

    np = numpy.array(cur_cut)
    slitscans(np, d)
    # thumbstrips(np, d)
    frames(np, d)
    moduloframes(cur_cut, d)    #takes cur_cut instead of np because needs timestamps
    hsv_metrics(np, d)
    features(np, d)

    return d

def analyze(path, height=96, min_n_frames=15, max_n_frames=3000):
    cur_cut = []
    cur_is_hard = False
    for frame in numm.video_frames(path, height=96, fps=25):
        if len(cur_cut)>min_n_frames and discontinuity(cur_cut[-1], frame, threshold=0.2):
            # HARD CUT
            yield cut_from_frames(cur_cut, cur_is_hard, frame.timestamp)
            cur_is_hard = True
            cur_cut = [frame]
        elif len(cur_cut)>min_n_frames and (len(cur_cut) >= max_n_frames or discontinuity(cur_cut[0], frame, threshold=0.4)):
            # SOFT CUT (or max_n_frames)
            yield cut_from_frames(cur_cut, cur_is_hard, frame.timestamp)
            cur_is_hard = False
            cur_cut = [frame]
        else:
            cur_cut.append(frame)
    yield cut_from_frames(cur_cut, cur_is_hard)

def serialize(videopath, directory=None):
    import os, pickle, time

    t0 = time.time()

    if directory is None:
        directory = videopath + '.analysis'
    absdir = os.path.join(directory, 'abs')

    if not os.path.isdir(directory):
        os.makedirs(directory)
    if not os.path.isdir(absdir):
        os.makedirs(absdir)

    doc = []
    for idx, c in enumerate(analyze(videopath)):
        subdir = os.path.join(directory, '%d' % (idx))
        if not os.path.isdir(subdir):
            os.makedirs(subdir)

        for name, im in c.images.iteritems():
            numm.np2image(im, os.path.join(subdir, '%s.png' % (name)))
        for name, np in c.arrays.iteritems():
            numpy.save(os.path.join(subdir,  '%s.npy' % (name)), np)

        for name, im in c.absimages.iteritems():
            numm.np2image(im, os.path.join(absdir, '%s.png' % (name)))

        doc.append(c.doc)

    pickle.dump(doc, open(os.path.join(directory, 'video.pickle'), 'w'))

    dur = doc[-1]['start'] + doc[-1]['duration']
    dt = time.time() - t0
    ratio = dur / dt

    print 'analyzed a %ds video in %ds (%.2fx)' % (dur, dt, ratio)


if __name__=='__main__':
    import sys
    serialize(sys.argv[1])

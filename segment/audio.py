import numm
import numpy

# configz
sample_rate=8000
windowsize =500
stepsize   =100

class Chunk(object):
    def __init__(self):
        self.doc = {}

def bark(fourier):
    # thanks, njoliat!
    BARK = [20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]
    # damn you, nyquist
    BARK = filter(lambda x: x<sample_rate/2, BARK)
    FREQS = numpy.fft.fftfreq(windowsize, 1.0/sample_rate)    
    BARK_FREQ = [(abs(FREQS - freq)).argmin() for freq in BARK]

    bark_spectrum = numpy.array([fourier[BARK_FREQ[i]:BARK_FREQ[i+1]].sum() \
                    for i in range(len(BARK_FREQ) - 1)])
    return bark_spectrum

def bark_spectogram(np):
    chunks = []
    cur_idx = 0
    while (cur_idx*stepsize + windowsize) < len(np):
        fourier = abs(numpy.fft.fft(np[cur_idx*stepsize:cur_idx*stepsize + windowsize]))
        chunks.append(bark(fourier))

        cur_idx += 1
    return numpy.array(chunks)

def segments(bark_spec, threshold=0.15, minlen=5):
    intensities = abs(bark_spec)

    # normalize each chunk
    normed = intensities - intensities.min(axis=1).reshape(intensities.shape[0], 1)
    normed = normed / normed.max(axis=1).reshape(normed.shape[0], 1)

    segs = numpy.ones(len(bark_spec), dtype=numpy.bool)

    diff = abs(normed[1:] - normed[:-1]).mean(axis=1)
    segs[1:] = diff > threshold

    # enforce minlen invarient
    for idx, s in enumerate(segs[:-minlen]):
        if s:
            segs[idx+1:idx+minlen] = False

    segment_indices = numpy.arange(len(segs))[segs]
    return segment_indices

def chunks(path):
    acc = numpy.zeros(0, numpy.int16)
    buffer_size = windowsize * 100
    start = 0.0                 # in seconds
    bark_spec = None

    # strategy:
    # accumulate mid-sized buffers, segment them, and then start the
    # next buffer from the last segment.

    for nbuf in numm.sound_chunks(path, sample_rate=sample_rate):
        acc = numpy.concatenate([acc, nbuf.mean(axis=1)])
        if len(acc) >= buffer_size:
            # 1. segment acc (yielding)
            bark_spec = bark_spectogram(acc)
            seg_indices = segments(bark_spec)
            for ii, start_idx in enumerate(seg_indices[:-1]): # ii == index index: eek!
                end_idx = seg_indices[ii+1]
                c = Chunk()
                c.doc['start'] = start + start_idx * stepsize / float(sample_rate)
                c.doc['duration'] = (end_idx - start_idx) * stepsize / float(sample_rate)
                c.doc['bark'] = bark_spec[start_idx:end_idx].mean(axis=0).tolist()
                c.doc['loudness'] = abs(acc[start_idx*stepsize:end_idx*stepsize]).mean() # rms? decibels? asdr?
                yield c
            # 2. reset acc to last segment
            if len(seg_indices)>0:
                acc = acc[seg_indices[-1]*stepsize:]
                # 3. & reset start
                start = start + (seg_indices[-1] * stepsize / float(sample_rate))

    # yield last segment
    # XXX: duplicated code
    c = Chunk()
    c.doc['start'] = start
    c.doc['duration'] = len(acc) / float(sample_rate)
    c.doc['bark'] = bark_spec.mean(axis=0).tolist()
    c.doc['loudness'] = abs(acc).mean()

def serialize(audiofile, directory=None):
    import os, pickle, time
    t0 = time.time()

    if directory is None:
        directory = audiofile + '.analysis'
    if not os.path.isdir(directory):
        os.makedirs(directory)

    doc = []
    for seg in chunks(audiofile):
        doc.append(seg.doc)

    dur = doc[-1]['start'] + doc[-1]['duration']
    dt = time.time() - t0
    ratio = dur / dt

    pickle.dump(doc, open(os.path.join(directory, 'audio.pickle'), 'w'))

    print 'analyzed a %ds audio file in %ds (%.2fx)' % (dur, dt, ratio)

if __name__=='__main__':
    import sys
    serialize(sys.argv[1])

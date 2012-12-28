import numm
import numpy as np

# defer on audio analysis -- just a waveform, for now

def _form(accum, h):
    arr = np.zeros((h, len(accum), 3), dtype=np.uint8)
    # normalize accum
    # XXX: this will lead to fake discontinuities at boundary!
    accum = np.array(accum)
    accum -= accum.min()
    accum /= accum.max()
    accum *= h
    for idx, val in enumerate(accum):
        pad = int((h - int(val))/2)
        arr[pad:-pad,idx,1] = 255
    return arr

def waveform(audiofile, fps=25, chunksize=500, height=96):
    # fps is video rate, s.t. one px of audio == one frame of video
    accum = []
    mult = float(height) / 2**12
    for c in numm.sound.precise_sound_chunks(audiofile, int(44100/fps)):
        accum.append(abs(c).mean())
        if len(accum) == chunksize:
            yield _form(accum, height)
            accum = []
    yield _form(accum, height)
        
def serialize(audiofile, directory=None):
    import os, time
    t0 = time.time()

    if directory is None:
        directory = audiofile + '.analysis'
    directory = os.path.join(directory, 'audio')
    if not os.path.isdir(directory):
        os.makedirs(directory)

    for idx,wform in enumerate(waveform(audiofile)):
        numm.np2image(wform, os.path.join(directory, "%06d.png" % (idx)))

    dt = time.time() - t0
    dur = idx * 20
    ratio = dur / dt
    print 'analyzed a ~%ds audio file in %ds (%.2fx)' % (dur, dt, ratio)

if __name__=='__main__':
    import sys
    serialize(sys.argv[1])

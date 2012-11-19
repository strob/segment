import glob
import json
import numpy as np
import os
import sys

for analysisdir in sys.argv[1:]:
    for npz in glob.glob(
        os.path.join(analysisdir, "*/*.npz")):
        payload = np.load(npz)
        out = {}
        for k,v in payload.items():
            out[k]=v.tolist()
        json.dump(out, open(npz.replace('.npz','.json'), 'w'))

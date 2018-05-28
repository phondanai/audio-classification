# coding= UTF-8
#
# Author: Fing
# Date  : 2017-12-03
#

import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import soundfile as sf
import sys
import scipy.io as sio
import h5py
import cv2


def load_eval_data():
    eval_data = None
    f = h5py.File('/home/phondanai/eval_feature.mat', mode='r')
    eval_data = f.get('evalFeatureCell').value[0]

    da = np.empty((13306,90))
    for i in eval_data:
        d = np.array(f[i])
        np.append(da, d, axis=0)

    #np.save('npy/eval_feat.npy', da)

    return da

def load_eval_label():
    tmp_eval = []
    with open('/home/phondanai/src/ASV/protocol/ASVspoof2017_eval_v2_key.trl.txt') as f:
       for i in f:
           tmp_eval.append(i.split()[1].strip())
    eval_keys = np.array([0 if i.split()[0].strip() == 'genuine' else 1 for i in tmp_eval ])

    return eval_keys


eval_feat = load_eval_data()
eval_label = load_eval_label()
np.save('eval_feat.npy', eval_feat)
np.save('eval_label.npy', eval_label)


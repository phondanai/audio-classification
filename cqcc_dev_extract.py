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


cqcc_features_dev_mat = sio.loadmat('/home/phondanai/src/Dev_feature.mat')
cqcc_features_dev = cqcc_features_dev_mat['devFeatureCell']

dev_label_dict = {}
with open('/home/phondanai/src/ASV/protocol/ASVspoof2017_dev.txt', 'r') as f:
    for line in f:
        f_name, label = line.split()[0].strip(), line.split()[1].strip()
        dev_label_dict[f_name] = 0 if label == 'genuine' else 1

dev_file_seq = {}
with open('/home/phondanai/src/ASV/protocol/ASVspoof2017_dev.txt', 'r') as f:
    for index, line in enumerate(f):
        k = line.split()[0].strip()
        dev_file_seq[k] = index

def extract_cqcc_dev(file_name):
    """
    Directly extract cqcc feature from Matlab .mat file, developmet dataset
    """

    f_key = file_name.split('/')[-1].strip()

    cqcc = np.mean(cqcc_features_dev[dev_file_seq[f_key]][0].T, axis=0)
    label = dev_label_dict[f_key]

    return cqcc, label


def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    #features, labels = np.empty((0,90)), np.empty(0) # 90 is cqcc
    features, labels = np.empty((0,84)), np.empty(0) # 84 is cqt
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                cqcc, label = extract_cqcc_dev(fn)
            except Exception as e:
                print("[Error] extract feature error. %s" % (e))
                continue
            ext_features = np.hstack(cqcc)
            features = np.vstack([features,ext_features])
            labels = np.append(labels, label)
        print("extract %s features done" % (sub_dir))
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

# Get features and labels
r = os.listdir("data/")
r.sort()
features, labels = parse_audio_files("data", r)

np.save('npy/cqt_dev_feat.npy', features)
np.save('npy/cqt_dev_label.npy', labels)


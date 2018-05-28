# coding= UTF-8
#
# Original Author: Fing
# Auhor : Phondanai Khanti
# Date  : 2018-05-28

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


cqcc_features_genuine = sio.loadmat('/home/phondanai/src/Genuine.mat')
cqcc_features_genuine['genuineFeatureCell']

cqcc_features_spoof = sio.loadmat('/home/phondanai/src/spoof.mat')
cqcc_features_spoof['spoofFeatureCell']


file_seq = {}
with open('/home/phondanai/src/ASV/protocol/ASVspoof2017_train.trn.txt', 'r') as f:
    for index, line in enumerate(f):
        k = line.split()[0].strip()
        file_seq[k] = index

label_dict = {}
with open('/home/phondanai/src/ASV/protocol/ASVspoof2017_train.trn.txt', 'r') as f:
    for line in f:
        f_name, label = line.split()[0].strip(), line.split()[1].strip()
        label_dict[f_name] = 0 if label == 'genuine' else 1


def extract_cqcc(file_name):
    """
    Directly extract cqcc feature from Matlab .mat file
    """
    f_key = file_name.split('/')[-1].strip()

    if file_seq[f_key] <= 1507:
        cqcc = np.mean(cqcc_features_genuine['genuineFeatureCell'][file_seq[f_key]][0].T, axis=0)
    else:
        new_key = file_seq[f_key] - 1508
        cqcc = np.mean(cqcc_features_spoof['spoofFeatureCell'][new_key][0].T, axis=0)
    label = label_dict[f_key]

    return cqcc, label


def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty((0,90)), np.empty(0) # 90 is cqcc
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                cqcc, label = extract_cqcc(fn)
            except Exception as e:
                print("[Error] extract feature error. %s" % (e))
                continue
            ext_features = np.hstack(cqt)
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
r = os.listdir('data/')
r.sort()
features, labels = parse_audio_files(data_dir, r)

np.save('train_feat.npy', features)
np.save('train_label.npy', labels)


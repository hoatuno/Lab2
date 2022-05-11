import librosa
import numpy as np
import os
import math
from sklearn.cluster import KMeans
import hmmlearn.hmm

def get_mfcc(file_path):
    y, sr = librosa.load(file_path) # read .wav file
#     hop_length = math.floor(sr*0.010) # 10ms hop
    hop_length = 256
#     win_length = math.floor(sr*0.025) # 25ms frame
    win_length = 512
    # mfcc is 12 x T matrix
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, n_fft=1024,
        hop_length=hop_length, win_length=win_length)
    # substract mean from mfcc --> normalize mfcc
#     mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1,1)) 
    mfcc = np.subtract(mfcc,np.mean(mfcc))
    # delta feature 1st order and 2nd order
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # X is 39 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
    # return T x 39 (transpose of X)
    return X.T # hmmlearn use T x N matrix

# test with file
a = get_mfcc('./14/19020066_HoangHuuTung/c15-c23.wav')
print(a.shape)

# extract all data

all_data = {}
all_labels = {}
for cname in class_names:
    file_paths = [os.path.join("./", cname, i) for i in os.listdir(os.path.join('./', cname)) if i.endswith('.wav')]
    data = []
    for file_path in file_paths:
        try:
            data.append(get_mfcc(file_path))
        except:
            print(file_path)
            os.remove(file_path) 
    
    all_data[cname] = data
    all_labels[cname] = [class_names.index(cname) for i in range(len(file_paths))]

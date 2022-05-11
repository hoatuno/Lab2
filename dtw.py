import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np
import os
from sklearn.metrics import classification_report
import random

def shuffler (arr, n):
     
    # We will Start from the last element
    # and swap one by one.
    for i in range(n-1,0,-1):
         
        # Pick a random index from 0 to i
        j = random.randint(0,i+1)
         
        # Swap arr[i] with the element at random index
        arr[i],arr[j] = arr[j],arr[i]
    return arr

# method extract
def normalize_mfcc(mfcc):
    for i in range(mfcc.shape[1]):
        mfcc[:, i] = mfcc[:, i] - np.mean(mfcc[:, i])
        mfcc[:, i] = mfcc[:, i] / np.max(np.abs(mfcc[:, i]))
    return mfcc


if __name__ == '__main__':
    lib_audios = {}
    template_audios = {}
    average_templates = {}
    test_labels = []
    class_names = ['xuong', 'len', 'phai',
                   'trai', 'nhay', 'ban', 'a', 'b', 'sil']
    for cname in class_names:
        file_paths = [os.path.join(
            "./", cname, i) for i in os.listdir(os.path.join('./', cname)) if i.endswith('.wav')]
        data = []
        index = 0
        for file_path in file_paths:
            try:
                data.append(librosa.load(file_path)[0])
                index = index + 1
                if index > 2:
                    break
            except:
                print(file_path)
                os.remove(file_path)
        lib_audios[cname] = data
    for cname in class_names:
        data = []
        for lib in lib_audios[cname]:
            data.append(normalize_mfcc(librosa.feature.mfcc(y=lib)))
            test_labels.append(cname)
        template_audios[cname] = data

    metric = 'cosine'
    for cname in class_names:
        cost01, align01 = librosa.sequence.dtw(
            template_audios[cname][0], template_audios[cname][1], metric=metric)
        cost02, align02 = librosa.sequence.dtw(
            template_audios[cname][0], template_audios[cname][2], metric=metric)
        count = np.ones(len(template_audios[cname][0][0]))
        summ = template_audios[cname][0].copy()
        for pair in align01:
            t, q = pair
            count[t] += 1
            summ[:, t] += template_audios[cname][1][:, q]

        for pair in align02:
            t, q = pair
            count[t] += 1
            summ[:, t] += template_audios[cname][2][:, q]

        average = summ / count
        average.shape
        average_templates[cname] = average

    # check sequence
    y_pred = []
    labels = []

    for i in range(0,len(test_labels)):
        cname_lib = test_labels[random.randint(0,len(test_labels)-1)]
        labels.append(cname_lib)
        cost = {}
        for cname_template in class_names:
            D, wp = librosa.sequence.dtw(normalize_mfcc(librosa.feature.mfcc(
                y=lib_audios[cname_lib][0])), average_templates[cname_template], metric=metric)
            
            cost[cname_lib] = D[normalize_mfcc(librosa.feature.mfcc(
                y=lib_audios[cname_lib][0])).shape[1]-1][average_templates[cname_template].shape[1]-1]
        pred=min(cost, key=lambda k: cost[k])
        y_pred.append(pred)

    report = classification_report(test_labels, y_pred, target_names=class_names)
    print(labels)
    print(y_pred)
    print(report)

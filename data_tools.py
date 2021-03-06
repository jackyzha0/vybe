from preprocess import features
import numpy as np
import glob
import os

def one_hot_encode(label):
    labels = np.zeros(5)
    labels[label] = 1
    return labels

def get_data(test=False):
    """
    Recursively find and return path names of all audio files and their labels
    Return a dictionary in this format
    {file path : label
    ...
    file path : label }
    """
    arr = np.loadtxt("/home/jacky/2kx/vybe/data/ESC-50-master/meta/esc50.csv",dtype=str,delimiter=',',skiprows=1)
    keydict = {}
    occ = [0,0,0,0,0]
    fourcount = 0
    if test:
        for i in range(len(arr)):
            if arr[i][0][0] == "5":
                if assign_num_to_label(arr[i][3]) == 4:
                    fourcount+=1
                if fourcount < 41 or assign_num_to_label(arr[i][3]) != 4:
                    key = 'data/ESC-50-master/audio/'+arr[i][0]
                    occ[assign_num_to_label(arr[i][3])] += 1
                    val = one_hot_encode(assign_num_to_label(arr[i][3]))
                    #print(val)
                    keydict.update({key:val})
        i = 0
    if not test:
        for i in range(len(arr)):
            if arr[i][0][0] != "5":
                if assign_num_to_label(arr[i][3]) == 4:
                    fourcount+=1
                if fourcount < 41 or assign_num_to_label(arr[i][3]) != 4:
                    key = 'data/ESC-50-master/audio/'+arr[i][0]
                    occ[assign_num_to_label(arr[i][3])] += 1
                    val = one_hot_encode(assign_num_to_label(arr[i][3]))
                    #print(val)
                    keydict.update({key:val})
        i = 0
    return keydict,len(keydict),occ

def assign_num_to_label(label):
    """
    Takes a string label and converts it to a number
    ESC-50 Key
    [0] Door Knock | ESC - 31
    [1] Clock Alarm | ESC - 38
    [2] Siren | ESC - 43, 8k - 7
    [3] Car Horn | ESC - 44, 8k - 1
    [4] Misc
    """
    if label == "door_wood_knock":
        return 0
    if label == "clock_alarm":
        return 1
    if label == "siren":
        return 2
    if label == "car_horn":
        return 3
    return 4

def next_minibatch(indices,db):
    """
    Return dictionary of next minibatch in this format
    (arr of mfcc) : label
    """
    feat = []
    lab = []
    #print(indices)
    for i in indices:
        #feat = features(db[i],13,parsePath=True)
        #print(i)
        z = db.keys()[i][25:-4]
        lab.append(db[db.keys()[i]])
        if os.path.exists('pickle/'+z+'.npy'):
            #print('Preloaded',z)
            l = np.load('pickle/'+z+'.npy')
            noise = np.random.normal(0, 0.05, l.shape)
            feat.append(l+noise)
        else:
            ftrtmp = np.empty((0,193))
            mfccs, chroma, mel, contrast,tonnetz=features(db.keys()[i])
            ext_mfccs = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            ftrtmp = np.vstack([ftrtmp,ext_mfccs])
            np.save('pickle/'+z,ftrtmp[0])
            noise = np.random.normal(0, 0.05, ftrtmp[0].shape)
            feat.append(ftrtmp[0]+noise)
            #print(np.asarray(feat).shape)
            print('Pickle saved to','pickle/'+z)
    # FEAT OUTPUT 501,26
    return np.asarray(feat),np.asarray(lab)

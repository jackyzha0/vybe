# !/usr/local/bin/python

def features(rawsnd,raw=False) :
    """Compute num amount of audio features of a sound
    Args:
        rawsnd : sound as array
        num : numbers of mfccs to compute
    Returns:
        Return a num x max_stepsize*32 feature vector
    """
    #import time
    #start = time.time()
    import librosa
    import numpy as np
    if not raw:
        X, sample_rate = librosa.load(rawsnd)
    else:
        X, sample_rate = (rawsnd,16000)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

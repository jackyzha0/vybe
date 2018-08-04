# !/usr/local/bin/python

def features(rawsnd, num, parsePath=False) :
    """Compute num amount of audio features of a sound
    Args:
        rawsnd : sound as array
        num : numbers of mfccs to compute
    Returns:
        Return a num x max_stepsize*32 feature vector
    """
    import time
    start = time.time()
    import librosa
    import numpy as np
    sample_rate=44100
    if parsePath:
        rawsnd, sample_rate = librosa.load(rawsnd, sr=16000)
    ft = librosa.feature.mfcc(y=rawsnd, sr=sample_rate, n_mfcc=num, n_fft=int(sample_rate*0.025), hop_length=int(sample_rate*0.010))
    ft[0] = librosa.feature.rmse(y=rawsnd, hop_length=int(0.010*sample_rate))
    deltas = librosa.feature.delta(ft)
    ft_plus_deltas = np.vstack([ft, deltas])
    ft_plus_deltas /= np.max(np.abs(ft_plus_deltas),axis=0)
    print(ft_plus_deltas.T.shape)
    end = time.time()
    print(end - start)
    return (ft_plus_deltas.T)

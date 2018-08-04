from preprocess import features
import numpy as np
import glob

def one_hot_encode(label):
    labels = np.zeros(5)
    labels[label] = 1
    print(labels)
    return labels

def get_data():
    """
    Recursively find and return path names of all audio files and their labels
    Return a dictionary in this format
    {file path : label
    ...
    file path : label }
    """
    arr = np.loadtxt("/home/jacky/2kx/vybe/data/ESC-50-master/meta/esc50.csv",dtype=str,delimiter=',',skiprows=1)
    keydict = {}
    for i in range(len(arr)):
        key = 'data/ESC-50-master/audio/'+arr[i][0]
        print(assign_num_to_label(arr[i][3]))
        val = one_hot_encode(assign_num_to_label(arr[i][3]))
        print(val)
        keydict.update({key:val})
    i = 0
    return keydict,len(keydict)

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
    if label == "glass_breaking":
        return 2
    if label == "door_wood_knock":
        return 3
    return 4

def next_minibatch(path_arr,batchsize):
    """
    Return dictionary of next minibatch in this format
    (arr of mfcc) : label
    """
    pass

def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Author: github.com/igormq
    Create a sparse representention of input array. For handling one-hot vector in targets
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)"""
    indices = []
    values = []
    for i, seq in enumerate(sequences):
        indices.extend(zip([i]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

def pad_sequences(sequences, maxlen=None, test=False,dtype=np.float32,
                  padding='post', truncating='post', value=0):
    '''
    Author: github.com/igormq
    Pads each sequence to the same length: the length of the longest
    sequence.
    If maxlen is provided, any sequence longer than maxlen is truncated to
    maxlen. Truncation happens off either the beginning or the end
    (default) of the sequence. Supports post-padding (default) and
    pre-padding.
    Args:
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger
        than maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
        lengths: numpy array with the original sequence lengths
    '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    if input_noise and not test:
        x += np.random.normal(scale=noise_magnitude,size=x.shape)
    return x, lengths

print(get_data())

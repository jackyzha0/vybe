import pyaudio
import time
import Queue
import numpy as np
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import scipy.signal as signal
import librosa.display
from preprocess import features

print('import ok')

CHANNELS = 2
RATE = 44100

p = pyaudio.PyAudio()
fulldata = np.array([])
dry_data = np.array([])
left = np.array([])
right = np.array([])
left_queue = Queue.Queue()
right_queue = Queue.Queue()

def main():
    global left_queue,right_queue
    stream = p.open(format=pyaudio.paFloat32,
        channels=CHANNELS,
        rate=RATE,
        output=False,
        input=True,
        stream_callback=callback)
    stream.start_stream()
    while stream.is_active():
        #21 cycles in a second
        #Listen for 3 sec
        #63 cycles
        #time.sleep(5)
        #stream.stop_stream()
        pass
    stream.close()

    left = np.hstack(list(left_queue.queue))
    right = np.hstack(list(right_queue.queue))
    # plt.plot(left)
    # plt.title("Left")
    # plt.show()
    # p.terminate()
    #
    # plt.plot(right)
    # plt.title("Right")
    # plt.show()
    # p.terminate()
    feat_left = features(left,13)
    feat_right = features(right,13)

    plt.plot(feat_left)
    plt.title("MFCC Left")
    librosa.display.specshow(feat_left, x_axis='time')
    plt.show()

    plt.plot(feat_right)
    plt.title("MFCC Right")
    librosa.display.specshow(feat_right, x_axis='time')
    p.terminate()

    plt.show()

def callback(in_data, frame_count, time_info, flag):
    global b,a,fulldata,dry_data,frames,left,right,left_queue,right_queue
    audio_data = np.fromstring(in_data, dtype=np.float32)
    #print('audio_data',audio_data.shape)
    dry_data = np.append(dry_data,audio_data)
    #do processing here
    fulldata = np.append(fulldata,audio_data)
    #print('fulldata',fulldata.shape)
    result = np.fromstring(audio_data, dtype=np.float32)
    #print('result0',result.shape)
    #print('result1',result.shape)
    #print(result[::2] == result[1::2])
    left = np.append(left,result[::2])
    right = np.append(right,result[1::2])
    left_queue.put(result[::2])
    right_queue.put(result[::2])
    while left_queue.qsize() > 63:
        for i in range(21):
            left_queue.get()
            right_queue.get()
    #print("left_q",left_queue.qsize())
    #print("right_q",right_queue.qsize())
    if left_queue.qsize() and right_queue.qsize() == 63:
        left_feat = features(np.hstack(list(left_queue.queue)),13)
        right_feat = features(np.hstack(list(right_queue.queue)),13)
    #print('left',left.shape)
    #print('right',right.shape)
    return (audio_data, pyaudio.paContinue)

main()

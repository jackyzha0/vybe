import pyaudio
import time
import Queue
import numpy as np
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import scipy.signal as signal
import librosa.display
import sugartensor as tf
from preprocess import features

print('import ok')

CHANNELS = 4
RATE = 16000

p = pyaudio.PyAudio()
fulldata = np.array([])
dry_data = np.array([])
left = np.array([])
right = np.array([])
top = np.array([])
bottom = np.array([])

left_feat = np.array([])
right_feat = np.array([])
top_feat = np.array([])
bottom_feat = np.array([])

left_queue = Queue.Queue()
right_queue = Queue.Queue()
top_queue = Queue.Queue()
bottom_queue = Queue.Queue()

feed_dict0 = {}
feed_dict1 = {}
feed_dict2 = {}
feed_dict3 = {}

graph_dir = "ckpt/model.meta"
savepath = "ckpt"
sess=tf.Session()
saver = tf.train.import_meta_graph(graph_dir)
saver.restore(sess, savepath+'/model')

graph = tf.get_default_graph()
op = sess.graph.get_operations()
#print(op)
x_inp = graph.get_tensor_by_name("input/x_inp:0")
out = graph.get_tensor_by_name("out/out:0")

def getHighestProb(arr):
    conf = 0
    j = 0
    s = 0
    for i in arr[0]:
        if i > conf:
            conf = i
            s = j
        j=j+1
    if s == 0:
        return "Door Knock - Confidence " + str(conf)
    if s == 1:
        return "Clock Alarm - Confidence " + str(conf)
    if s == 2:
        return "Siren - Confidence " + str(conf)
    if s == 3:
        return "Car Horn - Confidence " + str(conf)
    if s == 4:
        return "Misc. - Confidence " + str(conf)
def main():
    # imported_meta = tf.train.import_meta_graph("ckpt/model.meta")
    # with tf.Session() as session:
    #     imported_meta.restore(session, tf.train.latest_checkpoint('ckpt'))
    global left_feat,right_feat,top_feat,bottom_feat,feed_dict0,feed_dict1,feed_dict2,feed_dict3,x_inp,out,sess,graph
    stream = p.open(format=pyaudio.paFloat32,
        channels=CHANNELS,
        rate=RATE,
        output=False,
        input=True,
        stream_callback=callback)
    stream.start_stream()
    while stream.is_active():
        if left_queue.qsize() and right_queue.qsize() == 63:
            ftrtmp = np.empty((0,193))
            mfccs, chroma, mel, contrast,tonnetz = features(np.hstack(list(left_queue.queue)),raw=True)
            ext_mfccs = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            ftrtmp = np.vstack([ftrtmp,ext_mfccs])
            left_feat = ftrtmp[0]

            mfccs, chroma, mel, contrast,tonnetz = features(np.hstack(list(right_queue.queue)),raw=True)
            ext_mfccs = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            ftrtmp = np.vstack([ftrtmp,ext_mfccs])
            right_feat = ftrtmp[0]

            mfccs, chroma, mel, contrast,tonnetz = features(np.hstack(list(top_queue.queue)),raw=True)
            ext_mfccs = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            ftrtmp = np.vstack([ftrtmp,ext_mfccs])
            top_feat = ftrtmp[0]

            mfccs, chroma, mel, contrast,tonnetz = features(np.hstack(list(bottom_queue.queue)),raw=True)
            ext_mfccs = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            ftrtmp = np.vstack([ftrtmp,ext_mfccs])
            bottom_feat = ftrtmp[0]

            feed_dict0 = {x_inp:[left_feat]}
            feed_dict1 = {x_inp:[right_feat]}
            feed_dict2 = {x_inp:[top_feat]}
            feed_dict3 = {x_inp:[bottom_feat]}
            left = sess.run((out),feed_dict0)
            right = sess.run((out),feed_dict1)
            top = sess.run((out),feed_dict2)
            bottom = sess.run((out),feed_dict3)
            print('Left: '+getHighestProb(left))
            print('Right: '+getHighestProb(right))
            print('Top: '+getHighestProb(top))
            print('Bottom: '+getHighestProb(bottom))
        #21 cycles in a second
        #Listen for 3 sec
        #63 cycles
        #time.sleep(12.3)
        #stream.stop_stream()
        pass
    stream.close()

def callback(in_data, frame_count, time_info, flag):
    global inp,b,a,fulldata,dry_data,frames,left,right,top,bottom,left_feat,right_feat,top_feat,bottom_feat,left_queue,right_queue,top_queue,bottom_queue,feed_dict0,feed_dict1,feed_dict2,feed_dict3
    audio_data = np.fromstring(in_data, dtype=np.float32)
    #print('audio_data',audio_data.shape)
    dry_data = np.append(dry_data,audio_data)
    #do processing here
    fulldata = np.append(fulldata,audio_data)
    #print('fulldata',fulldata.shape)
    result = np.fromstring(audio_data, dtype=np.float32)
    result = np.reshape(result, (193, 4))
    #print('result0',result.shape)
    #print('result1',result.shape)
    #print(result[::2] == result[1::2])
    left_queue.put(result[:,0])
    right_queue.put(result[:,1])
    top_queue.put(result[:,2])
    bottom_queue.put(result[:,3])
    #print(left_queue.qsize())
    while left_queue.qsize() > 63:
        for i in range(21):
            left_queue.get()
            right_queue.get()
            top_queue.get()
            bottom_queue.get()
    return (audio_data, pyaudio.paContinue)

main()

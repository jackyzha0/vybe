print(time.strftime('[%H:%M:%S]'), 'Starting network... ')
import sugartensor as tf
print('[OK] tensorflow ')
import numpy as np
print('[OK] numpy ')
import os
import data_tools
print('[OK] os ')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config=tf.ConfigProto(allow_soft_placement=True)

dict_unprocessed = data_tools.get_data()
setsize = len(dict_unprocessed)
index = np.arrange(0,setsize)

### CLASS LIST ###
"""
ESC-50 Key
[0] Door Knock | ESC - 31
[1] Clock Alarm | ESC - 38
[2] Glass Breaking | ESC - 40
[3] Siren | ESC - 43, 8k - 7
[4] Car Horn | ESC - 44, 8k - 1
[5] Misc
"""

### PARAMETERS ###
batchsize = 128
num_classes = 6

### NN Setup
"""
Input Dims: 26 (features) x 147 (time length)
Output Dims: (num classes) x 147 (time length)
"""

with tf.device("/device:CPU:0"):
    graph = tf.Graph()
    with graph.as_deault():
        #ReLU Activations for all connections
        #M is number of feature maps

        #Input (F x T)
        #Convolution - Zero padding
        #Non-overlapping Frequency Max Pooling in only Frequency Domain
        #Convolution - Zero padding
        #Frequency Max Pooling (F' x M x T)

        #Stacking of Feature Maps of (F' x M x T)
        #RNN Activations (Sigmoid)
        #Feedforward Activations
        #Thresholding
        

### Training Loop

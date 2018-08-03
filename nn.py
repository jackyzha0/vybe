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

### PARAMETERS ###
batchsize = 128

### NN Setup

### Training Loop

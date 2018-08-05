import time
print('[OK] time')
print(time.strftime('[%H:%M:%S]'), 'Starting network... ')
import sugartensor as tf
print('[OK] tensorflow ')
import numpy as np
print('[OK] numpy ')
import os
import data_tools
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
print('[OK] os ')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config=tf.ConfigProto(allow_soft_placement=True)

### PARAMETERS ###
batchsize = 128
num_classes = 6
epochs = 100
learning_rate = 1e-4
num_features = 26
decay = 0.9
momentum = 0.9
num_channels = 2
frames = 501
kernel_size = 32
depth = 20
num_hidden = 200

### NN Setup
"""
Input Dims: 26 (features) x 147 (time length)
Output Dims: (num classes)
"""

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(1.0, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,2,2,1], padding='SAME')

def apply_convolution(x,kernel_size,num_channels,depth):
    weights = weight_variable([kernel_size, kernel_size, num_channels, depth])
    biases = bias_variable([depth])
    return tf.nn.relu(tf.add(conv2d(x, weights),biases))

def apply_max_pool(x,kernel_size,stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride_size, stride_size, 1], padding='SAME')

db,db_size = data_tools.get_data()
sd = 1/np.sqrt(num_features)
X  =  tf.placeholder(tf.float32, shape=[None,frames,num_features,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_classes])

cov = apply_convolution(X,kernel_size,num_channels,depth)

shape = cov.get_shape().as_list()
cov_flat = tf.reshape(cov, [-1, shape[1] * shape[2] * shape[3]])

f_weights = weight_variable([shape[1] * shape[2] * depth, num_hidden])
f_biases = bias_variable([num_hidden])
f = tf.nn.sigmoid(tf.add(tf.matmul(cov_flat, f_weights),f_biases))

out_weights = weight_variable([num_hidden, num_classes])
out_biases = bias_variable([num_classes])
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

cost_function = -tf.reduce_sum(Y * tf.log(y_))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
optimizer = tf.train.RMSPropOptimizer(learning_rate,decay=decay,momentum=momentum,centered=True).minimize(cost_function)
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history  = np.empty(shape=[1],dtype=float)

index = np.arange(db_size)
np.random.shuffle(index)

def get_indices(batchsize):
    global index
    if index.size < batchsize:
        index = np.arange(db_size)
        np.random.shuffle(index)
    ret = index[:batchsize]
    index = index[:-batchsize].copy()
    return ret

## Training Loop
cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None

with tf.Session() as session:
    tf.global_variables_initializer().run()

    for epoch in range(epochs):
        for batch in range(int(db_size / batchsize)):
            indices = get_indices(batchsize)
            feed = data_tools.next_minibatch(indices,db)
            print(feed[0].shape,feed[1].shape)
            _,cost = session.run([optimizer,cost_function],feed_dict={X: feed[0], Y: feed[1]})
            cost_history = np.append(cost_history,cost)

fig = plt.figure(figsize=(10,8))
plt.plot(cost_history)
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()


    # for itr in range(training_iterations):
    #     offset = (itr * batch_size) % (train_y.shape[0] - batch_size)
    #     batch_x = train_x[offset:(offset + batch_size), :, :, :]
    #     batch_y = train_y[offset:(offset + batch_size), :]
    #
    #     _, c = session.run([optimizer, cross_entropy],feed_dict={X: batch_x, Y : batch_y})
    #     cost_history = np.append(cost_history,c)
    #
    # #print('Test accuracy: ',round(session.run(accuracy, feed_dict={X: test_x, Y: test_y}) , 3))
    # fig = plt.figure(figsize=(15,10))
    # plt.plot(cost_history)
    # plt.axis([0,training_iterations,0,np.max(cost_history)])
    # plt.show()

# cost_function = -tf.reduce_sum(Y * tf.log(y_))

# correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#ReLU Activations for all connections
#M is number of feature maps

#Input (F x T)
#Convolution - Zero padding (3x3)
#Non-overlapping Frequency Max Pooling in only Frequency Domain
#Convolution - Zero padding (3x3)
#Frequency Max Pooling (F' x M x T)

#Stacking of Feature Maps of (F' x M x T)
#RNN Activations (Sigmoid)
#Feedforward Activations
#Thresholding

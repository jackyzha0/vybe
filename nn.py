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
batchsize = 4
num_classes = 5
epochs = 250
learning_rate = 1e-4
num_features = 193
n_hidden_units_one = 256
n_hidden_units_two = 512

### NN Setup
"""
Input Dims: 26 (features) x 501 (time length)
Output Dims: (num classes)
"""

db,db_size = data_tools.get_data()

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
sd = 1/np.sqrt(num_features)
X = tf.placeholder(tf.float32,[None, num_features])
Y = tf.placeholder(tf.float32,[None, num_classes])

W_1 = tf.Variable(tf.random_normal([ num_features,n_hidden_units_one], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)


W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)


W = tf.Variable(tf.random_normal([n_hidden_units_two, num_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([ num_classes], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)

init = tf.global_variables_initializer()

cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
#optimizer = tf.train.RMSPropOptimizer(learning_rate,decay=0.9,momentum=0.9,centered=True).minimize(cost_function)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1],dtype=float)
acc_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None

with tf.Session() as session:
    session.run(init)
    for epoch in range(epochs):
        for batch in range(int(db_size / batchsize)):
            indices = get_indices(batchsize)
            feed = data_tools.next_minibatch(indices,db)
            print("Batch",batch,"in epoch",epoch)
            _,cost,acc = session.run([optimizer,cost_function,accuracy],feed_dict={X: feed[0], Y: feed[1]})
            print("Cost: ",cost,"Accuracy: ",acc)
            cost_history = np.append(cost_history,cost)
            acc_history = np.append(acc_history,acc)

fig = plt.figure(figsize=(10,8))
plt.plot(cost_history)
plt.ylabel("Cost")
plt.xlabel("Epochs")
plt.axis([1,epochs,0,np.max(cost_history)])
plt.show()

plt.plot(acc_history)
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.axis([1,epochs,0,np.max(acc_history)])
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

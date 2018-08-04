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
epochs = 100
learningrate = 1e-4

### NN Setup
"""
Input Dims: 26 (features) x 147 (time length)
Output Dims: (num classes) x 147 (time length)
"""

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,num_classes])

W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two],
mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)

init = tf.initialize_all_variables()

cost_function = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
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


### Training Loop
cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        #second for loop for mini batching
        _,cost = sess.run([optimizer,cost_function],feed_dict={X:tr_features,Y:tr_labels})
        cost_history = np.append(cost_history,cost)

    y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: ts_features})
    y_true = sess.run(tf.argmax(ts_labels,1))
    print("Test accuracy: ",round(session.run(accuracy,
    	feed_dict={X: ts_features,Y: ts_labels}),3))

fig = plt.figure(figsize=(10,8))
plt.plot(cost_history)
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()

p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average="micro")
print "F-Score:", round(f,3)

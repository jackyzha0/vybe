import time
print('[OK] time')
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

### PARAMETERS ###
batchsize = 128
num_classes = 5
epochs = 1200
learning_rate = 0.005
num_features = 193
n_hidden_units_one = 256
n_hidden_units_two = 512
n_hidden_units_three = 1024

savepath = os.getcwd() + '/ckpt'


### NN Setup
"""
Input Dims: 26 (features) x 501 (time length)
Output Dims: (num classes)
"""

db,db_size,occ = data_tools.get_data()
t_db,t_db_size,_ = data_tools.get_data(test=True)
print(t_db_size)

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

def t_get_indices(batchsize):
    index = np.arange(batchsize)
    np.random.shuffle(index)
    return index

## Training Loop
sd = 1/np.sqrt(num_features)
with tf.name_scope('input'):
    X = tf.placeholder(tf.float32,[None, num_features],name="x_inp")
    Y = tf.placeholder(tf.float32,[None, num_classes],name="y_inp")

W_1 = tf.Variable(tf.random_normal([ num_features,n_hidden_units_one], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.tanh(tf.matmul(h_1,W_2) + b_2)

W_3 = tf.Variable(tf.random_normal([n_hidden_units_two,n_hidden_units_three], mean = 0, stddev=sd))
b_3 = tf.Variable(tf.random_normal([n_hidden_units_three], mean = 0, stddev=sd))
h_3 = tf.nn.sigmoid(tf.matmul(h_2,W_3) + b_3)

W = tf.Variable(tf.random_normal([n_hidden_units_three, num_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([num_classes], mean = 0, stddev=sd))
with tf.name_scope('out'):
    y_ = tf.nn.softmax(tf.matmul(h_3,W) + b,name="out")

init = tf.global_variables_initializer()

cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
#optimizer = tf.train.RMSPropOptimizer(learning_rate,decay=0.9,momentum=0.9,centered=True).minimize(cost_function)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1],dtype=float)
acc_history = np.empty(shape=[1],dtype=float)
t_cost_history = np.empty(shape=[1],dtype=float)
t_acc_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None

with tf.Session() as session:
    session.run(init)
    saver = tf.train.Saver()
    for epoch in range(epochs):
        for batch in range(int(db_size / batchsize)):
            indices = get_indices(batchsize)
            feed = data_tools.next_minibatch(indices,db)
            print("Batch",batch,"in epoch",epoch)
            _,cost,acc = session.run([optimizer,cost_function,accuracy],feed_dict={X: feed[0], Y: feed[1]})
            print("Cost: ",cost,"Accuracy: ",acc)
            cost_history = np.append(cost_history,cost)
            acc_history = np.append(acc_history,acc)
        t_indices = t_get_indices(64)
        t_feed = data_tools.next_minibatch(t_indices,t_db)
        _,t_cost,t_acc = session.run([optimizer,cost_function,accuracy],feed_dict={X: t_feed[0], Y: t_feed[1]})
        print("Test Cost: ",t_cost,"Test Accuracy: ",t_acc)
        t_cost_history = np.append(t_cost_history,t_cost)
        t_acc_history = np.append(t_acc_history,t_acc)

        save_path = saver.save(session, savepath+'/model')
        print(">>> Model saved succesfully")

# fig = plt.figure(figsize=(10,8))
# plt.plot(cost_history)
# plt.ylabel("Cost")
# plt.xlabel("Epochs")
# plt.axis([1,epochs,0,np.max(cost_history)])
# plt.show()
#
# plt.plot(acc_history)
# plt.ylabel("Accuracy")
# plt.xlabel("Epochs")
# plt.axis([1,epochs,0,np.max(acc_history)])
# plt.show()
#
# fig = plt.figure(figsize=(10,8))
# plt.plot(t_cost_history)
# plt.ylabel("Test Cost")
# plt.xlabel("Epochs")
# plt.axis([1,epochs,0,np.max(t_cost_history)])
# plt.show()
#
# plt.plot(t_acc_history)
# plt.ylabel("Test Accuracy")
# plt.xlabel("Epochs")
# plt.axis([1,epochs,0,np.max(t_acc_history)])
# plt.show()

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

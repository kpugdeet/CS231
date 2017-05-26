import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import math
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# clear old variables
tf.reset_default_graph()

# setup input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)

def simple_model(X, y):
    # define our weights (e.g. init_two_layer_convnet)
    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    W1 = tf.get_variable("W1", shape=[5408, 10]) #Stride = 2, ((32-7)/2)+1 = 13, 13*13*32=5408
    b1 = tf.get_variable("b1", shape=[10])

    # define our graph (e.g. two_layer_convnet)
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1, 2, 2, 1], padding='VALID') + bconv1 # Stride [batch, height, width, channels]
    h1 = tf.nn.relu(a1)
    h1_flat = tf.reshape(h1, [-1, 5408]) # Flat the output to be size 5408 each row
    y_out = tf.matmul(h1_flat, W1) + b1
    return y_out


y_out = simple_model(X, y)

# define our loss
total_loss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=y_out)
mean_loss = tf.reduce_mean(total_loss)

# define our optimizer
optimizer = tf.train.AdamOptimizer(5e-4)  # select optimizer and set learning rate
train_step = optimizer.minimize(mean_loss)


def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False, dropOut=1.0):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [loss_val, correct_prediction, accuracy]
    if training_now:
        variables[-1] = training

    # counter
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
            # generate indicies for the batch
            start_idx = (i * batch_size) % Xd.shape[0]
            idx = train_indicies[start_idx:start_idx + batch_size]

            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx, :],
                         y: yd[idx],
                         is_training: training_now, # we don't have to use training_now
                         keep_prob: dropOut}
            # get batch size
            actual_batch_size = yd[idx].shape[0]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables, feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss * actual_batch_size)
            correct += np.sum(corr)

            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}".format(iter_cnt, loss, np.sum(corr) / float(actual_batch_size)))
            iter_cnt += 1
        total_correct = correct / float(Xd.shape[0])
        total_loss = np.sum(losses) / Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}".format(total_loss, total_correct, e + 1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e + 1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.savefig("Tensorflow.png")
            plt.clf()
    return total_loss, total_correct


with tf.Session() as sess:
    with tf.device("/cpu:0"):  # "/cpu:0" or "/gpu:0"
        sess.run(tf.global_variables_initializer())
        print('Training')
        run_model(sess, y_out, mean_loss, X_train, y_train, 1, 64, 100, train_step, True)
        print('Validation')
        run_model(sess, y_out, mean_loss, X_val, y_val, 1, 64)

# clear old variables
tf.reset_default_graph()

# define our input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)

print "\n\n############## Complex Model ###################"
# define model
def complex_model(X,y,is_training):
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32]) # Stride = 2, ((32-7)/2)+1 = 13, 13*13*32=5408, NO padding
    bconv1 = tf.get_variable("bconv1", shape=[32])
    scale1 = tf.Variable(tf.ones([32]))
    beta1 = tf.Variable(tf.zeros([32]))
    W1 = tf.get_variable("W1", shape=[1152, 1024])  # After MaxPool, No padding, (13/2)**2 * 32 = 1152
    b1 = tf.get_variable("b1", shape=[1024])
    W2 = tf.get_variable("W2", shape=[1024, 10])
    b2 = tf.get_variable("b2", shape=[10])


    # define our graph (e.g. two_layer_convnet)
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1, 2, 2, 1], padding='VALID') + bconv1  # Stride [batch, height, width, channels]
    h1 = tf.nn.relu(a1)
    batch_mean1, batch_var1 = tf.nn.moments(h1, [0, 1, 2]) # Spatial batch over batch height width
    bn1 = tf.nn.batch_normalization(h1, batch_mean1, batch_var1, beta1, scale1, 1e-3)
    mp1 = tf.nn.max_pool(bn1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    mp1_flat = tf.reshape(mp1, [-1, 1152])  # Flat the output to be size 5408 each row
    y_out1 = tf.matmul(mp1_flat, W1) + b1
    y_outRelu = tf.nn.relu(y_out1)
    y_out2 = tf.matmul(y_outRelu, W2) + b2
    return y_out2
    pass

y_out = complex_model(X,y,is_training)

# Now we're going to feed a random batch into the model
# and make sure the output is the right size
x = np.random.randn(64, 32, 32,3)
with tf.Session() as sess:
    with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0"
        tf.global_variables_initializer().run()
        a = time.time()
        ans = sess.run(y_out,feed_dict={X:x,is_training:True})
        print (time.time() - a)
        print(ans.shape)
        print(np.array_equal(ans.shape, np.array([64, 10])))

try:
    with tf.Session() as sess:
        with tf.device("/gpu:0") as dev: #"/cpu:0" or "/gpu:0"
            tf.global_variables_initializer().run()
            a = time.time()
            ans = sess.run(y_out,feed_dict={X:x,is_training:True})
            print (time.time() - a)
except tf.errors.InvalidArgumentError:
    print("no gpu found, please use Google Cloud if you want GPU acceleration")
    # rebuild the graph
    # trying to start a GPU throws an exception
    # and also trashes the original graph
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)
    y_out = complex_model(X,y,is_training)


# Inputs
#     y_out: is what your model computes
#     y: is your TensorFlow variable with label information
# Outputs
#    mean_loss: a TensorFlow variable (scalar) with numerical loss
#    optimizer: a TensorFlow optimizer
# This should be ~3 lines of code!
# mean_loss = None
# optimizer = None
total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y, 10), logits=y_out)
mean_loss = tf.reduce_mean(total_loss)
optimizer = tf.train.RMSPropOptimizer(1e-3)  # select optimizer and set learning rate
pass


# batch normalization in tensorflow requires this extra dependency (For batch_mean1, batch_var1)
# Ensures that we execute the update_ops before performing the train_step
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('./tensorflow', sess.graph)
print('Training')
run_model(sess,y_out,mean_loss,X_train,y_train,1,64,100,train_step)


print('Validation')
run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
writer.close()

#Train a great model on CIFAR-10!
# Feel free to play with this cell
print "\n\n############## My Model ###################"
def my_model(X,y,is_training):
    # Cov1: Stride = 1, ((32-5))+1 = 28, 28*28*32, No padding
    Wconv1 = tf.get_variable("Wconv1", shape=[5, 5, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1, 1, 1, 1],padding='VALID') + bconv1  # Stride [batch, height, width, channels]

    # BN1
    scale1 = tf.Variable(tf.ones([32]))
    beta1 = tf.Variable(tf.zeros([32]))
    batch_mean1, batch_var1 = tf.nn.moments(a1, [0, 1, 2])
    bn1 = tf.nn.batch_normalization(a1, batch_mean1, batch_var1, beta1, scale1, 1e-3)

    # Relu1
    h1 = tf.nn.relu(bn1)

    # Drop1
    h1_drop = tf.nn.dropout(h1, keep_prob)  # DropOut

    # Cov2: Stride = 1, ((28-5))+1 = 24, 24*24*64, No padding
    Wconv2 = tf.get_variable("Wconv2", shape=[5, 5, 32, 64])
    bconv2 = tf.get_variable("bconv2", shape=[64])
    a2 = tf.nn.conv2d(h1_drop, Wconv2, strides=[1, 1, 1, 1], padding='VALID') + bconv2  # Stride [batch, height, width, channels]

    # BN2
    scale2 = tf.Variable(tf.ones([64]))
    beta2 = tf.Variable(tf.zeros([64]))
    batch_mean2, batch_var2 = tf.nn.moments(a2, [0, 1, 2])
    bn2 = tf.nn.batch_normalization(a2, batch_mean2, batch_var2, beta2, scale2, 1e-3)

    # Relu2
    h2 = tf.nn.relu(bn2)

    # Drop2
    h2_drop = tf.nn.dropout(h2, keep_prob)  # DropOut

    # Pool1, No padding, (24/2)=12, 12*12*64
    mp1 = tf.nn.max_pool(h2_drop, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # FC1
    W1 = tf.get_variable("W1", shape=[9216, 1024])
    b1 = tf.get_variable("b1", shape=[1024])
    mp1_flat = tf.reshape(mp1, [-1, 9216])  # Flat the output to be size 9216 each row
    y_out1 = tf.matmul(mp1_flat, W1) + b1

    # BN3
    scale3 = tf.Variable(tf.ones([1024]))
    beta3 = tf.Variable(tf.zeros([1024]))
    batch_mean3, batch_var3 = tf.nn.moments(y_out1, [0])
    bn3 = tf.nn.batch_normalization(y_out1, batch_mean3, batch_var3, beta3, scale3, 1e-3)

    y_outRelu1 = tf.nn.relu(bn3)

    # FC2
    W2 = tf.get_variable("W2", shape=[1024, 10])
    b2 = tf.get_variable("b2", shape=[10])
    y_out = tf.matmul(y_outRelu1, W2) + b2

    return y_out
    pass

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)

y_out = my_model(X,y,is_training)
# mean_loss = None
# optimizer = None
total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y, 10), logits=y_out)
mean_loss = tf.reduce_mean(total_loss)
# optimizer = tf.train.RMSPropOptimizer(1e-3)  # select optimizer and set learning rate

pass

# Weight Decay every 1000 steps with a base of 0.96:
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 500, 0.95, staircase=True)

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = (tf.train.AdamOptimizer(learning_rate).minimize(mean_loss, global_step=global_step))


# Feel free to play with this cell
# This default code creates a session
# and trains your model for 10 epochs
# then prints the validation set accuracy
sess = tf.Session()

sess.run(tf.global_variables_initializer())
print('Training')
run_model(sess,y_out,mean_loss,X_train,y_train,10,64,100,train_step,True, 0.5)
print('Validation')
run_model(sess,y_out,mean_loss,X_val,y_val,1,64)



# Test your model here, and make sure
# the output of this cell is the accuracy
# of your best model on the training and val sets
# We're looking for >= 70% accuracy on Validation
print('Training')
run_model(sess,y_out,mean_loss,X_train,y_train,1,64)
print('Validation')
run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
print('Test')
run_model(sess,y_out,mean_loss,X_test,y_test,1,64)
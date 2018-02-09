import tensorflow as tf
import numpy as np
from datetime import datetime

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

n_input = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_output = 10

X = tf.placeholder(tf.float32, shape=(None, n_input), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

def writeLog(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def neuron_layer(X, num_neuron, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        std = 2/np.sqrt(n_inputs) # originally, it is np.sqrt. So what is the difference?
        init = tf.truncated_normal((n_inputs, num_neuron), stddev=std)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([num_neuron]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        return Z

with tf.name_scope("dnn"):
    drop_X = tf.layers.dropout(X, rate=0.5)
    hidden1 = neuron_layer(drop_X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    drop_Hidden1 = tf.layers.dropout(hidden1, rate=0.5)
    hidden2 = neuron_layer(drop_Hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    drop_Hidden2 = tf.layers.dropout(hidden2, rate=0.5)
    logits = neuron_layer(hidden2, n_output, name="output")

# with tf.name_scope("dnn"): # the best way is to use tensorflow official document
#     hidden1 = tf.layers.dense(X, n_hidden1, "hidden1", activation=tf.nn.relu)
#     hidden2 = tf.layers.dense(hidden1, n_hidden2, "hidden2", activation=tf.nn.relu)
#     logits = tf.layers.dense(hidden2, n_output, "output")
#
# with tf.name_scope("loss"):
#     xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
#     loss = tf.reduce_mean(xentropy, name = "loss")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    writeLog(xentropy)
    loss = tf.reduce_mean(xentropy, name="loss")
    writeLog(loss)

learn_rate = 0.01

with tf.name_scope("train"):
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learn_rate, momentum=0.9, name="momentum")
    optimizer = tf.train.MomentumOptimizer(learning_rate=learn_rate, momentum=0.9, use_nesterov=True, name="NesterovWithDropout")
    # optimizer = tf.train.GradientDescentOptimizer(learn_rate, name="Gradient")
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learn_rate, momentum=0.9, decay=0.9, epsilon=1e-10, name="RMS") # performs far worse than the rest
    # optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate, name="ADAM")
    name_Opt = optimizer.get_name()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

now = datetime.utcnow().strftime("%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}{}".format(root_logdir, now, name_Opt)

n_epoches = 20
batch_size = 30

merged = tf.summary.merge_all()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epoches):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        summary, acc_train = sess.run([merged, accuracy], feed_dict={X: X_batch, y: y_batch})
        acc_val = sess.run(accuracy, feed_dict={X: mnist.validation.images, y: mnist.validation.labels})

        file_writer.add_summary(summary, epoch*batch_size)
        print("Epoch", epoch, "Train Accuracy", acc_train, "Test Accuracy", acc_val)
    save_path = saver.save(sess, "./final_model.ckpt")

file_writer.close()
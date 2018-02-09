import tensorflow as tf
from datetime import datetime
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

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

n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10

learning_rate = 0.01
# learning_rate = 0.001

X = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs], name="X")
y = tf.placeholder(tf.int32, shape=[None], name="y")

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits = tf.layers.dense(states, n_outputs, activation=tf.nn.softmax)

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
writeLog(xentropy)
loss = tf.reduce_mean(xentropy)
writeLog(loss)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.5, use_nesterov=True, name="UseNesterov")
name_Opt = optimizer.get_name()
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

now = datetime.utcnow().strftime("%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}{}".format(root_logdir, now, name_Opt)

merged = tf.summary.merge_all()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
saver = tf.train.Saver()

n_epoches = 100
batch_size = 150

init = tf.global_variables_initializer()

with tf.Session() as sess:
    saver.restore(sess, "./final_model.ckpt")
    sess.run(init)
    for epoch in range(1, n_epoches):
        for iteration in range(1, mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        eval_accuracy, summary = sess.run([accuracy, merged], feed_dict={X: X_batch, y: y_batch})
        test_images = mnist.test.images.reshape((-1, n_steps, n_inputs))
        acc_test = sess.run(accuracy, feed_dict={X: test_images, y: mnist.test.labels})
        file_writer.add_summary(summary, epoch * batch_size)
        print(epoch, "eval: ", eval_accuracy, "test: ", acc_test)
    save_path = saver.save(sess, "./final_model.ckpt")
file_writer.close()
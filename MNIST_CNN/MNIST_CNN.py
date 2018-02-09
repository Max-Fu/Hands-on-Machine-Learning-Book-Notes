import tensorflow as tf
import numpy as np
from datetime import datetime

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

print(mnist.train.images.shape)

n_input = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_output = 10

n_epoches = 20
batch_size = 30

X = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
training=True
X_stage2 = tf.placeholder(tf.float32, shape=(None, n_input), name="X_2")

# X = tf.placeholder(tf.float32, shape=(None, n_input), name="X")
# y = tf.placeholder(tf.int64, shape=(None), name="y")
training = True

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

with tf.name_scope("CNN"):
    # drop_X = tf.layers.dropout(X, rate=0.5, training=training)
    net_layer1 = tf.layers.conv2d(X, filters=32, kernel_size=[5,5], strides=[2,2], padding="SAME", activation=tf.nn.relu, name="layer1")
    max_pool1 = tf.layers.max_pooling2d(net_layer1, pool_size=[2, 2], strides=2, padding="SAME")
    net_layer2 = tf.layers.conv2d(max_pool1, filters=64, kernel_size=[5,5], strides=[2,2], padding="SAME", activation=tf.nn.relu, name="layer2")
    max_pool2 = tf.layers.max_pooling2d(net_layer2, pool_size=[2, 2], strides=2, padding="SAME")
    pool2_flat = tf.reshape(max_pool2, [-1, 256])
    drop_X = tf.layers.dropout(pool2_flat, rate=0.5)
    fc1 = tf.contrib.layers.fully_connected(drop_X, 84)
    drop_fc1 = tf.layers.dropout(fc1, rate=0.5)
    fc2 = tf.contrib.layers.fully_connected(drop_fc1, 50)
    drop_fc2 = tf.layers.dropout(fc2, rate=0.5)
    logits = tf.contrib.layers.fully_connected(drop_fc2, n_output)
    print("logits", logits.shape)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    writeLog(xentropy)
    loss = tf.reduce_mean(xentropy, name="loss")
    writeLog(loss)

learn_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learn_rate, momentum=0.9, use_nesterov=True, name="NesterovWithDropout")
    name_Opt = optimizer.get_name()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    print(logits, y)
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

now = datetime.utcnow().strftime("%Y%m%D%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}{}".format(root_logdir, now, name_Opt)


merged = tf.summary.merge_all()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epoches):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = np.reshape(X_batch, [-1, 28, 28, 1])
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        summary, acc_train = sess.run([merged, accuracy], feed_dict={X: X_batch, y: y_batch})
        X_acc_val = np.reshape(mnist.validation.images, [-1, 28, 28, 1])
        acc_val = sess.run(accuracy, feed_dict={X: X_acc_val, y: mnist.validation.labels})

        file_writer.add_summary(summary, epoch*batch_size)
        print("Epoch", epoch, "Train Accuracy", acc_train, "Test Accuracy", acc_val)
    save_path = saver.save(sess, "./final_model.ckpt")

file_writer.close()
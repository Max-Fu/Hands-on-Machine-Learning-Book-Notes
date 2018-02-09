from functools import partial
import tensorflow as tf
import tensorflow.contrib as learn

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")

n_inputs = 28*28
n_layer1 = 300
n_layer2 = 150
n_layer3 = 300
n_ouputs = n_inputs

l2_reg = 0.0001
learning_rate = 0.001

X = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs])

he_init = learn.layers.variance_scaling_initializer()
reg = learn.layers.l2_regularizer(l2_reg)

dense_layer = partial(tf.layers.dense,
                      activation=tf.nn.elu,
                      use_bias=True,
                      kernel_initializer=he_init,
                      kernel_regularizer=reg
                      )

layer1 = dense_layer(X, n_layer1)
layer2 = dense_layer(layer1, n_layer2)
layer3 = dense_layer(layer2, n_layer3)
outputs = learn.layers.fully_connected(layer3, n_ouputs)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_epochs = 20
batch_size = 50
n_iterations = 50

with tf.Session() as sess:
    sess.run(init)
    for epochs in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_iterations):
            X_, y_ = mnist.train.next_batch(n_batches)
            sess.run(training_op, feed_dict={X: X_})
            # discard, loss = sess.run([training_op, loss], feed_dict={X: X_})
        print("epoch, ", epochs)
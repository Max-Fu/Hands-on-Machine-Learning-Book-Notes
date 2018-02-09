import tensorflow as tf
import tensorflow.contrib as learn
from functools import partial
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden2
n_hidden4 = n_hidden1
n_outputs = n_inputs

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

l2 = 0.0001
learning_rate = 0.001

he_init = learn.layers.variance_scaling_initializer()
l2_init = learn.layers.l2_regularizer(l2)

# dense_layers = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=he_init, kernel_regularizer=l2_init)

layer1 = tf.layers.dense(X, n_hidden1, name="layer1", activation=tf.nn.elu, kernel_initializer=he_init, kernel_regularizer=l2_init)
layer2 = tf.layers.dense(layer1, n_hidden2, name="layer2", activation=tf.nn.elu, kernel_initializer=he_init, kernel_regularizer=l2_init)
layer3 = tf.layers.dense(layer2, n_hidden3, name="layer3", activation=tf.nn.elu, kernel_initializer=he_init, kernel_regularizer=l2_init)
layer4 = tf.layers.dense(layer3, n_hidden4, name="layer4", activation=tf.nn.elu, kernel_initializer=he_init, kernel_regularizer=l2_init)
output = learn.layers.fully_connected(layer4, n_outputs)


# layer1 = dense_layers(X, n_hidden1)
# layer2 = dense_layers(layer1, n_hidden2)
# layer3 = dense_layers(layer2, n_hidden3)
# layer4 = dense_layers(layer2, n_hidden4)
# output = dense_layers(layer4, n_outputs, activation=None)

# xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=X)
loss = tf.reduce_mean(tf.square(output-X)) #MSE

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

difference = tf.square(output-X)

init = tf.global_variables_initializer()

n_epochs = 100
n_iterations = 30
batch_size = 30
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        # for iteration in range(n_iterations):
        #     X_batch, y_batch = mnist.train.next_batch(batch_size)
        #     sess.run(training_op, feed_dict={X: X_batch})
        X_batch, y_batch = mnist.train.next_batch(batch_size)
        nn_ouput, difference = sess.run([training_op, difference], feed_dict={X: X_batch})
        print("Epoch: ", epoch, "difference, ", difference)


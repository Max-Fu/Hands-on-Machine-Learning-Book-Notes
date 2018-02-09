import tensorflow as tf

import numpy as np

X0_batch = np.array([[1,2,3], [3,4,5], [1,2,3], [3,4,5], [1,2,3]])
X1_batch = np.array([[4,5,6], [6,7,8], [4,5,6], [6,7,8], [4,5,6]])
X_batch = np.array([
    [[1,2,3], [3,4,5]],
    [[1,2,3], [3,4,5]],
    [[1,2,3], [1,2,3]]
])

X_batch_sequence = np.array([
    [[1,2,3], [3,4,5]],
    [[1,2,3], [0,0,0]],
    [[1,2,3], [1,2,3]]
])

n_input = 3
n_neuron = 5
n_neuron_dynamic = 3

#static RNN

# X0 = tf.placeholder(tf.float32, shape=[None, n_input], name="X0")
# X1 = tf.placeholder(tf.float32, shape=[None, n_input], name="X1")
#
# basic_rnn = tf.contrib.rnn.BasicRNNCell(num_units=n_neuron)
# all, last_stage = tf.contrib.rnn.static_rnn(basic_rnn, [X0, X1], dtype=tf.float32)
# Y0, Y1 = all
# # W0 = tf.Variable(tf.random_normal([n_input, n_neuron], dtype=tf.float32))
# # W1 = tf.Variable(tf.random_normal([n_neuron, n_neuron], dtype=tf.float32))
# # bias = tf.Variable(tf.zeros([1, n_neuron], dtype=tf.float32))
# #
# # Y0 = tf.tanh(tf.matmul(X0, W0)+bias)
# # Y1 = tf.tanh(tf.matmul(Y0, W1)+tf.matmul(X1, W0)+bias)
#
# init = tf.global_variables_initializer()


# Dynamic RNN
# n_steps = 2
# X = tf.placeholder(tf.float32, [None, n_steps, n_neuron_dynamic])
# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neuron_dynamic)
# output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)


# Dynamic RNN with Variable Length
n_steps = 2
seq_length = tf.placeholder(tf.int32, [None])
X = tf.placeholder(tf.float32, [None, n_steps, n_neuron_dynamic])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neuron_dynamic)
output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32, sequence_length=seq_length)

#Session:

init = tf.global_variables_initializer()
sequence_length_batch = np.array([2,1,2])
with tf.Session() as sess:
    sess.run(init)
    output_eval, output_state = sess.run([output, states], feed_dict={X: X_batch, seq_length: sequence_length_batch})
    print(output_eval)


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

test_size = 0.2

test_size = 0.2

t = np.random.sample((1000, 100))
y_with_noise = t * np.sin(t) / 3 + 2 * np.sin(5 * t)

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

def next_batch(batch_size):
    random_list = np.random.choice(1000, batch_size)
    return t[random_list].reshape(-1, n_steps, n_inputs), y_with_noise[random_list].reshape(-1, n_steps, n_inputs)

X = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs], name="X")
y = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs], name="y")

cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.relu), output_size=n_outputs)
# cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu), output_size=n_outputs)
outputs, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
# rnn_outputs, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
# stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_inputs])
# stacked_outputs = tf.layers.dense(stacked_rnn_outputs, units=n_outputs)
# outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_inputs])


learning_rate = 0.001
loss = tf.reduce_mean(tf.square(outputs-y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.5, use_nesterov=True)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

mean_loss = tf.reduce_mean(loss)

average_difference = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(outputs, y))))

n_iterations = 2000
batch_size = 50

iteration_total = []
average_difference_list = []
aver_mean_loss = []

with tf.Session() as sess:
    init.run()
    for iteration in range(1, n_iterations):
        X_batch, y_batch = next_batch(batch_size)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        X_batch_Eval, y_batch_Eval = next_batch(batch_size=20)
        average_difference_value, mean_loss_value = sess.run([average_difference, mean_loss], feed_dict={X: X_batch_Eval, y: y_batch_Eval})
        print(iteration, "eval_accuracy: ", average_difference_value, "eval loss:", mean_loss_value)
        average_difference_list.append(average_difference_value)
        aver_mean_loss.append(mean_loss_value)
        iteration_total.append(iteration)

plt.plot(iteration_total, average_difference_list)
plt.savefig("./aver_difference_with_adam.png")

plt.plot(iteration_total, aver_mean_loss)
plt.savefig("./aver_mean_loss_with_adam.png")


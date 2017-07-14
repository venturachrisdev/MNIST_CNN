import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

""" Data and hyperparameters """

# Training hyperp
config = {
	# Training epocs
	'epocs': 20000,
	# Gradient Descent learning rate (alpha /or/ eta)
	'learning_rate': 1e-4,
	# [num, width, height, channels]
	'image_shape': [-1, 28, 28, 1],
	# [num, width * height * channels]
	'flatimage_shape': [-1, 784],
}

# Dataset
mnist = read_data_sets("mnist", one_hot=True)
# Split dataset
X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

# helpers

def new_weight(shape, name):
	""" Create random weights given a shape"""
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

def new_bias(shape, name):
	""" Create fixed biases given a shape """
	return tf.Variable(tf.constant(0.1, shape=shape), name=name)

# Tensorflow Graph - Model

x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y = tf.placeholder(tf.float32, shape=[None, 10], name="y")
# Image matrix (28 x 28 x 1)
x_image = tf.reshape(x, config["image_shape"], name="reshape_x_image")

"""
----------- First layer [Convolutional - ReLU - Pooling ] ----------
"""
# conv layer (5 x 5 x 1) with 32 filters
W_conv1 = new_weight([5, 5, 1, 32], name="W_conv1")
# One bias for each (32) filter
b_conv1 = new_bias([32], name="b_conv1")
# Apply each filter to the convoluted layer with stride of 1 and 0 padding
z_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding="SAME", name="conv2d_z_conv1")
# Apply ReLU to each filter (replace negatives with zero)
h_conv1 = tf.nn.relu(z_conv1 + b_conv1, "relu_h_conv1")
# Max pooling (downsampling) 2x2
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="max_pool_h_pool1")


"""
----- Second layer [ Convolutional - ReLU - Pooling ] ----------
"""
# Weights for 64 5x5 filters given 32 inputs
W_conv2 = new_weight([5, 5, 32, 64], name="W_conv2")
# One bias for each (64) filter.
b_conv2 = new_bias([64], name="b_conv2")
# Apply each filter to the convoluted layer with stride of 1 and 0 padding
z_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding="SAME", name="conv2d_z_conv2")
# Apply ReLU to each filter (replace negatives with zero)
h_conv2 = tf.nn.relu(z_conv2 + b_conv2, name="relu_h_conv2")
# Max pooling (downsampling) 2x2
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="max_pool_h_pool2")

"""
---- Thrid layer [ Fully-connected ] -------
"""
W_fc1 = new_weight([7 * 7 * 64, 1024], name="W_fc1")
b_fc1 = new_bias([1024], name="b_fc1")

h_pool2_flattered = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name="reshape_hpool2_flattered")
z_fc1 = tf.matmul(h_pool2_flattered, W_fc1, name="matmul_z_fc1")
h_fc1 = tf.nn.relu(z_fc1 + b_fc1, name="relu_h_fc1")

"""
---- Four layer [ Fully-connected ] -------
"""
W_fc2 = new_weight([1024, 10], name="W_fc2")
b_fc2 = new_bias([10], name="b_fc2")

z_fc2 = tf.matmul(h_fc1, W_fc2, name="matmul_z_fc2")

y_pred = tf.nn.softmax(z_fc2 + b_fc2, name="y_pred")

"""
--- Backprop and learning -----------
"""

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
optimizer = tf.train.AdamOptimizer(config['learning_rate'])
train_step = optimizer.minimize(cross_entropy)
correct_pred = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Train
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for e in range(config['epocs']):
		X_batch, y_batch = mnist.train.next_batch(50)
		train_step.run(feed_dict={x: X_batch, y: y_batch})
		if e % 500 == 0:
			print("Step: %d ------------" % e)
			loss = cross_entropy.eval(feed_dict={x: X_batch, y: y_batch})
			print("\tLoss: %.2f" % loss)
		if e % 1000 == 0:
			saver.save(sess, './mnist_tf_conv_model/mnist_tf_conv_model', global_step=e)
			# test_acc = accuracy.eval(feed_dict={x: X_test, y: y_test})
			# print("\tAccuracy: %.2f" % test_acc)
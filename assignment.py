from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random

EPOCHS = 10



class Colorizer(tf.keras.Model):
	def __init__(self):
		"""
    This model class will contain the architecture for your CNN that 
		classifies images. Do not modify the constructor, as doing so 
		will break the autograder. We have left in variables in the constructor
		for you to fill out, but you are welcome to change them if you'd like.
		"""
		super(Model, self).__init__()

		# Initialize all hyperparameters
		self.optimizer = optimizer = tf.keras.optimizers.Adam(learning_rate=)
		self.learning_rate1 = .00003
		self.learning_rate2 = .00001
		self.learning_rate3 = 0.000003

		# LAB Colorscheme constants
		self.num_a_partitions = 20
		self.num_b_partitions = 20
		self.a_min = -86.185
		self.a_max = 98.254
		self.b_min = -107.863
		self.b_max = 94.482
		self.num_classes = self.num_a_partitions * self.num_b_partitions
		self.a_range = self.a_max - self.a_min
		self.b_range = self.b_max - self.b_min
		self.a_class_size = self.a_range / self.num_a_partitions
		self.b_class_size = self.b_range / self.num_b_partitions

		# Initialize all trainable parameters
		self.relu = tf.nn.relu()
		self.bn = tf.keras.layers.BatchNormalization()
		self.conv1_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, dilation_rate=1, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv1_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, dilation_rate=1, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv2_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, dilation_rate=1, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv2_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, dilation_rate=1, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv3_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, dilation_rate=1, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv3_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, dilation_rate=1, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv3_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, dilation_rate=1, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv4_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv4_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv4_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv5_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv5_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv5_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv6_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv6_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv6_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv7_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv7_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv7_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv8_1 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding="SAME", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv8_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="SAME", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
		self.conv8_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=1, padding="SAME", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))

		self.conv8_num_classes = tf.keras.layers.Conv2D(filters=self.num_classes, kernel_size=1, strides=1, dilation_rate=1, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))

	def normalize(inputs):
		mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2])
		return tf.nn.batch_normalization(inputs, mean, variance, None, None, 0.001)

	def temp_softmax(self, z):
		"""
		Calculates the f_T function on a single pixel.
		:param z: a array representing the probability of each pixel being in different color bins. Size (num_bins)
		:return: the output of the modified softmax, dimension: (num_bins)
		"""
		z_exponent = tf.math.exp(tf.math.log(z) / self.T)
		z_sum = tf.reduce_sum(z_exponent)
		return z_exponent / z_sum

	def h_function(self, image):
		width = image.shape[0]
		height = image.shape[0][0]
		output = tf.Variable(shape=(width, height, 2))
		for w in range(width):
			for h in range(height):
				output[w][h] = self.h_function_pixel(output[w][h])
		return output

	def h_function_pixel(self, pixel):
		"""
		Calculates the h function for a single pixel.
		:param pixel: a array representing the probability of each pixel being in different color bins. Size (num_bins)
		:return: the expectation of a and b. Size 1x2: (a,b)
		"""
		expectation_a = 0
		expectation_b = 0
		prob = self.temp_softmax(z)
		for i in range(pixel.shape[0]):
			a, b = self.bin_to_ab(i)
			# prob values should all be less than 1
			expectation_a = a * prob[i] + expectation_a
			expectation_b = b * prob[i] + expectation_b
		return expectation_a,expectation_b

	def ab_to_bin(self, a, b):
		a_index = int((a - self.a_min) / self.a_range)
		b_index = int((b - self.b_min) / self.b_range)
		bin_num = a_index * self.num_a_partitions + b_index
		return bin_num

	def bin_to_ab(self, bin_id):
		"""
		Sets up a dictionary that maps from bin number to ab values
		:param bin_val: bin number (which is 0-indexed)
		:return: corresponding ab value as a tuple (a, b)
		"""
		a_index = bin_id / self.num_a_partitions
		b_index = bin_id % self.num_a_partitions
		a = (a_index + 0.5) * self.a_range + self.a_min
		b = (b_index + 0.5) * self.b_range + self.b_min
		return a,b

	def call(self, inputs):
		"""
		Runs a forward pass on an input batch of images.
		:param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
		:return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
		"""
		output1 = self.normalize(self.relu(self.conv1_2(self.relu(self.conv1_1(inputs)))))
		output2 = self.normalize(self.relu(self.conv2_2(self.relu(self.conv2_1(output1)))))
		output3 = self.normalize(self.relu(self.conv3_3(self.relu(self.conv3_2(self.relu(self.conv3_1(output2)))))))
		output4 = self.normalize(self.relu(self.conv4_3(self.relu(self.conv4_2(self.relu(self.conv4_1(output3)))))))
		output5 = self.normalize(self.relu(self.conv5_3(self.relu(self.conv5_2(self.relu(self.conv5_1(output4)))))))
		output6 = self.normalize(self.relu(self.conv6_3(self.relu(self.conv6_2(self.relu(self.conv6_1(output5)))))))
		output7 = self.normalize(self.relu(self.conv7_3(self.relu(self.conv7_2(self.relu(self.conv7_1(output6)))))))
		output8 = self.relu(self.conv8_3(self.relu(self.conv8_2(self.relu(self.conv8_1(output7))))))
		y_hat = self.conv8_num_classes(output8)
		return self.h_function(y_hat)

	def loss(self, logits, labels):
		"""
		Calculates the model cross-entropy loss after one forward pass.
		:param logits: during training, a matrix of shape (batch_size, self.num_classes) 
		containing the result of multiple convolution and feed forward layers
		Softmax is applied in this function.
		:param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
		:return: the loss of the model as a Tensor
		"""

		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))

	def accuracy(self, logits, labels):
		"""
		Calculates the model's prediction accuracy by comparing
		logits to correct labels – no need to modify this.
		:param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
		containing the result of multiple convolution and feed forward layers
		:param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

		NOTE: DO NOT EDIT
		
		:return: the accuracy of the model as a Tensor
		"""
		correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
		return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
	"""
	Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
	and labels - ensure that they are shuffled in the same order using tf.gather.
	To increase accuracy, you may want to use tf.image.random_flip_left_right on your
	inputs before doing the forward pass. You should batch your inputs.
	:param model: the initialized model to use for the forward pass and backward pass
	:param train_inputs: train inputs (all inputs to use for training),
	shape (num_inputs, width, height, num_channels)
	:param train_labels: train labels (all labels to use for training),
	shape (num_labels, num_classes)
	:return: None
	"""
	num_examples = train_inputs.shape[0]
	indices = range(num_examples - 1)
	indices = tf.random.shuffle(indices)
	tf.gather(train_inputs, indices)
	tf.gather(train_labels, indices)
	train_inputs = tf.image.random_flip_left_right(train_inputs)
	i = 0
	min_index = i * model.batch_size
	while min_index < num_examples:
		max_index = min_index + model.batch_size
		if max_index >= num_examples:
			max_index = num_examples - 1
		inputs = train_inputs[min_index:max_index, :, :, :]
		labels = train_labels[min_index:max_index]
		with tf.GradientTape() as tape:
			predictions = model.call(inputs)
			loss = model.loss(predictions, labels)
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
		i += 1
		min_index = i * model.batch_size
	loss = model.loss(predictions, labels)
	print("Loss: " + str(loss))
	return


def test(model, test_inputs, test_labels):
	"""
	Tests the model on the test inputs and labels. You should NOT randomly 
	flip images or do any extra preprocessing.
	:param test_inputs: test data (all images to be tested), 
	shape (num_inputs, width, height, num_channels)
	:param test_labels: test labels (all corresponding labels),
	shape (num_labels, num_classes)
	:return: test accuracy - this can be the average accuracy across 
	all batches or the sum as long as you eventually divide it by batch_size
	"""
	test_logits = model.call(test_inputs, True)
	return model.accuracy(test_logits, test_labels)
	pass

def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
	"""
	Uses Matplotlib to visualize the results of our model.
	:param image_inputs: image data from get_data(), limited to 10 images, shape (10, 32, 32, 3)
	:param probabilities: the output of model.call(), shape (10, num_classes)
	:param image_labels: the labels from get_data(), shape (10, num_classes)
	:param first_label: the name of the first class, "dog"
	:param second_label: the name of the second class, "cat"

	NOTE: DO NOT EDIT

	:return: doesn't return anything, a plot should pop-up 
	"""
	predicted_labels = np.argmax(probabilities, axis=1)
	num_images = image_inputs.shape[0]

	fig, axs = plt.subplots(ncols=num_images)
	fig.suptitle("PL = Predicted Label\nAL = Actual Label")
	for ind, ax in enumerate(axs):
			ax.imshow(image_inputs[ind], cmap="Greys")
			pl = first_label if predicted_labels[ind] == 0.0 else second_label
			al = first_label if np.argmax(image_labels[ind], axis=0) == 0 else second_label
			ax.set(title="PL: {}\nAL: {}".format(pl, al))
			plt.setp(ax.get_xticklabels(), visible=False)
			plt.setp(ax.get_yticklabels(), visible=False)
			ax.tick_params(axis='both', which='both', length=0)
	plt.show()


def main():
	'''
	Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
	test your model for a number of epochs. We recommend that you train for
	10 epochs and at most 25 epochs. For CS2470 students, you must train within 10 epochs.
	You should receive a final accuracy on the testing examples for cat and dog of >=70%.
	:return: None
	'''
	class1 = 3 # usually 3 - cat
	class2 = 5 # usually 5 - dog
	training_inputs, training_labels = get_data("CIFAR_data_compressed/train", class1, class2)
	test_inputs, test_labels = get_data("CIFAR_data_compressed/test", class1, class2)
	model = Model()

	for ep in range(EPOCHS):
		print("Epoch: " + str(ep + 1))
		train(model, training_inputs, training_labels)
	acc = test(model, test_inputs, test_labels)
	print("FINAL ACCURACY: %.3f" % acc)
	probabilities = model.call(test_inputs)
	start = tf.random.uniform([], dtype=tf.dtypes.int32, maxval=test_inputs.shape[0] - 11)
	visualize_results(test_inputs[start:start + 9, :, :, :], probabilities[start:start + 9], test_labels[start:start + 9], "cat", "dog")
	return


if __name__ == '__main__':
	main()

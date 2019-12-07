from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from skimage import color

import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Activation
from tensorflow.keras.initializers import RandomNormal
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
		super(Colorizer, self).__init__()

		# Initialize all hyperparameters
		self.batch_size = 64
		self.learning_rate1 = .00003
		self.learning_rate2 = .00001
		self.learning_rate3 = 0.000003
		self.temperature = 0.38
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate1)

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
		self.bin_to_ab_arr = np.zeros(shape=(400, 2), dtype=np.float32)
		self.stdev = .0313

		# Initialize all trainable parameters
		self.model = tf.keras.Sequential()
		#section 1
		self.model.add(Conv2D(filters=64, kernel_size=3, strides=1, dilation_rate=1, padding="same",
							  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=64, kernel_size=3, strides=2, dilation_rate=1, padding="same",
							  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(BatchNormalization())

		#section 2
		self.model.add(Conv2D(filters=128, kernel_size=3, strides=1, dilation_rate=1, padding="same",
											  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=128, kernel_size=3, strides=2, dilation_rate=1, padding="same",
											  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(BatchNormalization())

		#section 3
		self.model.add(Conv2D(filters=256, kernel_size=3, strides=1, dilation_rate=1, padding="same",
											  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=256, kernel_size=3, strides=1, dilation_rate=1, padding="same",
											  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=256, kernel_size=3, strides=2, dilation_rate=1, padding="same",
											  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(BatchNormalization())

		#section 4
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same",
											  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same",
											  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same",
											  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(BatchNormalization())


		#section 5
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2, padding="same",
											  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2, padding="same",
											  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2, padding="same",
											  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(BatchNormalization())

		#section 6
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2, padding="same",
											  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2, padding="same",
											  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2, padding="same",
											  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(BatchNormalization())

		#section 7
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same",
											  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same",
											  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same",
											  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(BatchNormalization())

		#section 8
		self.model.add(Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding="SAME",
													   kernel_initializer=RandomNormal(
														   stddev=0.1), activation='relu'))
		self.model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="SAME",
											  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=256, kernel_size=4, strides=1, padding="SAME",
											  kernel_initializer=RandomNormal(stddev=self.stdev), activation='relu'))
		#self.model.add(BatchNormalization())

		#section 9
		self.model.add(Conv2DTranspose(filters=256, kernel_size=4, strides=4, padding="SAME",
													   kernel_initializer=RandomNormal(
														   stddev=self.stdev), activation='relu'))

		#y_hat (convert num classes)
		self.model.add(Conv2D(filters=self.num_classes, kernel_size=1, strides=1,
														dilation_rate=1, padding="same",
														kernel_initializer=RandomNormal(
															stddev=self.stdev)))

	def normalize(inputs):
		mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2])
		return tf.nn.batch_normalization(inputs, mean, variance, None, None, 0.001)

	def h_function(self, image):
		width = image.shape[1]
		height = image.shape[2]
		#output = np.zeros((image.shape[0], width, height, 2))

		probs = tf.nn.softmax(image / self.temperature)
		ab = tf.tensordot(probs, self.bin_to_ab_arr, axes=((3), (0)))
		return ab



		# for i in range(output.shape[0]):
		# 	for w in range(width):
		# 		for h in range(height):
		# 			a, b = self.h_function_pixel(image[i][w][h])
		# 			output[i][w][h][0] = a
		# 			output[i][w][h][1] = b
		# return output

	# def h_function_pixel(self, pixel):
	# 	"""
	# 	Calculates the h function for a single pixel.
	# 	:param pixel: a array representing the probability of each pixel being in different color bins. Size (num_bins)
	# 	:return: the expectation of a and b. Size 1x2: (a,b)
	# 	"""
	# 	expectation_a = 0
	# 	expectation_b = 0
	# 	prob = tf.nn.softmax(tf.div(pixel, self.temperature))
	#
	# 	for i in range(pixel.shape[0]):
	# 		a, b = self.bin_to_ab(i)
	# 		# prob values should all be less than 1
	# 		expectation_a = a * prob[i] + expectation_a
	# 		expectation_b = b * prob[i] + expectation_b
	# 	return expectation_a, expectation_b

	def ab_to_bin(self, a, b):
		a_index = int((a - self.a_min) / self.a_range)
		b_index = int((b - self.b_min) / self.b_range)
		bin_num = a_index * self.num_a_partitions + b_index
		return bin_num

	def bin_to_ab(self, bin_id):
		"""
		Calculates a, b values for a given bin id
		:param bin_val: bin number (which is 0-indexed)
		:return: corresponding ab value as a tuple (a, b)
		"""
		a_index = bin_id / self.num_a_partitions
		b_index = bin_id % self.num_a_partitions
		a = (a_index + 0.5) * self.a_range + self.a_min
		b = (b_index + 0.5) * self.b_range + self.b_min
		return a, b

	def init_bin_to_ab_array(self):
		"""
		Fill in the values for the bin_to_ab array
		"""
		for i in range(self.bin_to_ab_arr.shape[0]):
			a, b = self.bin_to_ab(i)
			self.bin_to_ab_arr[i][0] = a
			self.bin_to_ab_arr[i][1] = b

	def call(self, inputs):
		"""
		Runs a forward pass on an input batch of images.
		:param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
		:return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
		"""
		y_hat = self.model(inputs)
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
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels[:, :, :, 1:3], logits))

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

	# random_indices = np.arange(len(train_inputs))
	# random_indices = tf.random.shuffle(random_indices)
	# inputs = tf.gather(train_inputs, random_indices)
	# labels = tf.gather(train_labels, random_indices)
	# inputs = tf.image.random_flip_left_right(inputs)
	#
	# with tf.GradientTape() as tape:
	# 	predictions = model(inputs)
	# loss = tf.reduce_mean(model.loss(predictions, labels))
	#
	# gradients = tape.gradient(loss, model.trainable_variables)
	# model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

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
	print("Loss: {}".format(loss))
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

def visualize_images(bw_images, color_images, predictions):
	num_images = bw_images.shape[0]

	fig, axs = plt.subplots(nrows=3, ncols=num_images)
	fig.suptitle("Images\n ")
	reformatted = np.zeros([bw_images.shape[0], bw_images.shape[1], bw_images.shape[2], 3])
	reformatted_predictions = np.zeros([bw_images.shape[0], bw_images.shape[1], bw_images.shape[2], 3])
	for i in range(bw_images.shape[0]):
		for w in range(bw_images.shape[1]):
			for h in range(bw_images.shape[2]):
				reformatted[i, w, h, 0] = bw_images[i, w, h, 0]
				reformatted_predictions[i, w, h, 0] = bw_images[i, w, h, 0]
				reformatted_predictions[i, w, h, 0] = predictions[i, w, h, 0]
				reformatted_predictions[i, w, h, 0] = predictions[i, w, h, 1]
	for ind, ax in enumerate(axs):
		for i in range(len(ax)):
			a = ax[i]
			if ind == 0:
				a.imshow(color.lab2rgb(reformatted[i]), cmap="Greys")
				a.set(title="BW")
			elif ind == 1:
				a.imshow(color.lab2rgb(color_images[i]), cmap="Greys")
				a.set(title="Real")
			else:
				a.imshow(color.lab2rgb(reformatted_predictions[i]), cmap="Greys")
				a.set(title="Predicted")
			plt.setp(a.get_xticklabels(), visible=False)
			plt.setp(a.get_yticklabels(), visible=False)
			a.tick_params(axis='both', which='both', length=0)
	plt.show()


def main():
	'''
	Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
	test your model for a number of epochs. We recommend that you train for
	10 epochs and at most 25 epochs. For CS2470 students, you must train within 10 epochs.
	You should receive a final accuracy on the testing examples for cat and dog of >=70%.
	:return: None
	'''

	training_inputs, training_labels = get_data('../CIFAR_data_compressed/train')
	test_inputs, test_labels = get_data('../CIFAR_data_compressed/test')

	model = Colorizer()
	epochs = 1
	batch_size = 64
	batch = 0

	for e in range(epochs):
		batch_start = 0
		while (batch_start + batch_size) < len(training_inputs):
			batch += 1
			batch_end = batch_start + batch_size
			if batch_end > len(training_inputs):
				batch_end = len(training_inputs)
			train(model, training_inputs[batch_start:batch_end, :, :, :], training_labels[batch_start:batch_end, :, :, :])
			print("Epoch: {}/{} Batch: {}/{} Accuracy: {}".format(e + 1, epochs, batch, len(training_inputs)/batch_size, model.accuracy(model.call(training_inputs[batch_start:batch_end, :, :, :]), training_labels[batch_start:batch_end, :, :, 1:3])))
			batch_start += batch_size
	print("Testing!")
	print("Final Accuracy: {}".format(test(model, test_inputs, test_labels)))
	model.init_bin_to_ab_array()

	predictions = model.call(test_inputs[0:5, :, :])

	visualize_images(test_inputs[0:5, :, :], test_labels[0:5, :, :], predictions)

	return


if __name__ == '__main__':
	main()

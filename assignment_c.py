# -*- coding: utf-8 -*-
from __future__ import absolute_import
import matplotlib
# comment out the following if running on a computer and trying to display final image visualizer
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from preprocess_c import get_data
from skimage import color

import os
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Activation
from tensorflow.keras.initializers import RandomNormal, TruncatedNormal
import numpy as np
import random
import argparse
from scipy.stats import norm
from scipy.ndimage import gaussian_filter


gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)

# we used argparse to make defining hyperparameters easier. Running this with no parameters tests it, use --mode train to train
# to get full accuracy on full test set use --full-test TRUE, by default we just visualize and do accuracy on 5 images
parser = argparse.ArgumentParser(description='Colorizer')

parser.add_argument('--mode', type=str, default='test', help='Can be "train" or "test"')
parser.add_argument('--device', type=str, default='GPU:0' if gpu_available else 'CPU:0',
                    help='specific the device of computation eg. CPU:0, GPU:0, GPU:1, GPU:2, ... ')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Sizes of image batches fed through the network')
parser.add_argument('--num-epochs', type=int, default=10,
                    help='Number of passes through the training data to make before stopping')
parser.add_argument('--display', type=bool, default=False,
                    help='False saves file, True displays')
parser.add_argument('--full-test', type=bool, default=False,
                    help='True gets final accuracy')
parser.add_argument('--bin-init-images', type=int, default=100,
                    help='Number of images to initialize the bin distribution on')
parser.add_argument('--bin-init-batch-size', type=int, default=10,
                    help='Batch size for initializing the bin distribution on')
parser.add_argument('--skip-bin-init', type=bool, default=False,
                    help='Will load bin distribution from checkpoints instead for training if true')

args = parser.parse_args()

if not args.display:
	matplotlib.use('Agg')

class Colorizer(tf.keras.Model):
	def __init__(self):
		"""
    	This This defines the model for our colorizer. It mostly consists of sets of convolution layers.
		"""
		super(Colorizer, self).__init__()

		# Initialize all hyperparameters
		self.learning_rate1 = .00003
		self.learning_rate2 = .00001
		self.learning_rate3 = 0.000003
		self.temperature = 0.38
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate1)

		# LAB Colorscheme constants
		self.num_a_partitions = 20
		self.num_b_partitions = 20
		self.a_min = tf.constant(-86.185, dtype=tf.float32)
		self.a_max = tf.constant(98.254, dtype=tf.float32)
		self.b_min = tf.constant(-107.863, dtype=tf.float32)
		self.b_max = tf.constant(94.482, dtype=tf.float32)
		self.num_classes = self.num_a_partitions * self.num_b_partitions
		self.a_range = tf.constant(self.a_max - self.a_min, dtype=tf.float32)
		self.b_range = tf.constant(self.b_max - self.b_min, dtype=tf.float32)
		self.a_class_size = self.a_range / tf.dtypes.cast(self.num_a_partitions, tf.float32)
		self.b_class_size = self.b_range / tf.dtypes.cast(self.num_b_partitions, tf.float32)

		# Bin constants
		self.bin_to_ab_arr = self.init_bin_to_ab_array()
		self.expansion_size = 0.0001
		self.stdev = 0.04
		self.bin_distribution = tf.zeros(shape=[self.num_classes], dtype=tf.float32)
		self.bin_distance_stddev = 5
		self.gaussian_filter_stddev = 5
		self.lam = .5
		self.w = tf.zeros(shape=[self.num_classes], dtype=tf.float32)
		# .0313

		# Initialize all trainable parameters
		self.model = tf.keras.Sequential()
		# section 1
		self.model.add(Conv2D(filters=64, kernel_size=3, strides=1, dilation_rate=1, padding="same",
							  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=64, kernel_size=3, strides=2, dilation_rate=1, padding="same",
							  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(BatchNormalization())

		# section 2
		self.model.add(Conv2D(filters=128, kernel_size=3, strides=1, dilation_rate=1, padding="same",
											  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=128, kernel_size=3, strides=2, dilation_rate=1, padding="same",
											  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(BatchNormalization())

		# section 3
		self.model.add(Conv2D(filters=256, kernel_size=3, strides=1, dilation_rate=1, padding="same",
											  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=256, kernel_size=3, strides=1, dilation_rate=1, padding="same",
											  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=256, kernel_size=3, strides=2, dilation_rate=1, padding="same",
											  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(BatchNormalization())

		# section 4
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same",
											  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same",
											  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same",
											  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(BatchNormalization())


		# section 5
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2, padding="same",
											  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2, padding="same",
											  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2, padding="same",
											  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(BatchNormalization())

		# section 6
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2, padding="same",
											  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2, padding="same",
											  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2, padding="same",
											  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(BatchNormalization())

		# section 7
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same",
											  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same",
											  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same",
											  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(BatchNormalization())

		# section 8
		self.model.add(Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding="SAME",
													   kernel_initializer=TruncatedNormal(
														   stddev=0.1), activation='relu'))
		self.model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="SAME",
											  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(Conv2D(filters=256, kernel_size=4, strides=1, padding="SAME",
											  kernel_initializer=TruncatedNormal(stddev=self.stdev), activation='relu'))
		self.model.add(BatchNormalization())

		#section 9
		self.model.add(Conv2DTranspose(filters=256, kernel_size=4, strides=4, padding="SAME",
													   kernel_initializer=TruncatedNormal(
														   stddev=self.stdev), activation='relu'))
		self.model.add(BatchNormalization())

		#y_hat (convert num classes)
		self.model.add(Conv2D(filters=self.num_classes, kernel_size=1, strides=1,
														dilation_rate=1, padding="same",
														kernel_initializer=TruncatedNormal(
															stddev=self.stdev), activation='softmax'))
		self.init_bin_to_ab_array()


	def h_function(self, image):
		"""
		Maps from predicted bin space to ab space
		:param: image
		:return: corresponding bin id
		"""
		#Calculate probabilities and dot with the conversion from bin to ab summing over pixels
		probs = tf.nn.softmax(image / self.temperature)
		ab = tf.tensordot(probs, self.bin_to_ab_arr, axes=((3), (0)))
		return ab

	def ab_to_bin(self, a, b):
		"""
		Calculates the bin id that an a,b pair falls into
		:param a color component
		:param b color component
		:return: corresponding bin id
		"""
		#calculate fraction of ab space through which this falls
		a_index = (a - self.a_min) / self.a_range
		b_index = (b - self.b_min) / self.b_range
		#Multiply by partition size and sum to create bin space
		bin_num = int(tf.dtypes.cast(a_index, tf.float32) * self.num_a_partitions * self.num_b_partitions + tf.dtypes.cast(b_index, tf.float32) * self.num_b_partitions)
		return bin_num

	def bin_to_ab(self, bin_id):
		"""
		Calculates a, b values for a given bin id
		:param bin_val: bin number (which is 0-indexed)
		:return: corresponding ab value as a tuple (a, b)
		"""
		#Calculate the index for a and b
		a_index = tf.dtypes.cast(tf.dtypes.cast(bin_id, tf.float32) / self.num_a_partitions, tf.float32)
		b_index = tf.dtypes.cast(tf.dtypes.cast(bin_id, tf.float32) % self.num_a_partitions, tf.float32)
		#Center the index and shift by min value
		a = (a_index + 0.5) / self.num_a_partitions * self.a_range + self.a_min
		b = (b_index + 0.5) / self.num_b_partitions * self.b_range + self.b_min
		return tf.convert_to_tensor(a), tf.convert_to_tensor(b)

	def init_bin_to_ab_array(self):
		"""
		Fill in the values for the bin_to_ab array
		"""
		#Create a temp array for storage
		temp_arr = np.zeros(shape=(self.num_a_partitions * self.num_b_partitions, 2), dtype=np.float32)
		#Fill the array with ab values
		for i in range(temp_arr.shape[0]):
			a, b = self.bin_to_ab(i)
			temp_arr[i][0] = a
			temp_arr[i][1] = b
		return tf.convert_to_tensor(temp_arr, dtype=tf.float32)

	def init_w(self):
		"""
		Initializes self.w field based on the formula from the paper (page 6, equation 4)
		"""
		#Using the formula on page 6, we form, weight, and normalize w
		w = np.reciprocal(self.bin_distribution * (1 - self.lam) + self.lam / self.num_classes)
		unbalanced_expectation = np.dot(self.bin_distribution, w)
		#Gaussian filter is applied to the output
		self.w = gaussian_filter(w / unbalanced_expectation, sigma=self.gaussian_filter_stddev)

	def v(self, z):
		"""
		Weights for less common bins (page 6, equation 3)
		:param z, (batch_size, h, w, q) dimension, bin distribution for all images in a batch
		:return: weight to multiply by for the most likely bin
		"""
		#Get the max position in the distribution and return the value in w at that position
		pos = tf.math.argmax(z)
		return self.w[pos]

	def loss(self, predictions, labels):
		"""
		Calculates loss between probability distribution of bins for real and predicted image (page 5, equation 2)
		:param labels, (batch_size, h, w, 2) dimension matrix of real image's ab values
		:param predictions, (batch_size, h, w, q) dimension matrix of predicted
		:return: loss
		"""
		#Calculate the distribution for the labels
		z = self.calculate_bin_distribution(labels)
		num_images = predictions.shape[0]
		h = predictions.shape[1]
		w = predictions.shape[2]
		#Set up a blank v array
		v_blank = np.zeros(shape=(predictions.shape[:3]))
		#Form the right-most summation from the paper
		summation_1 = tf.keras.losses.categorical_crossentropy(z, predictions)
		#Form the v values for all pixels
		for x in range(num_images):
			for i in range(h):
				for j in range(w):
					v_blank[x,i,j] = self.v(z[x,i,j,:])
		#Dot over them to form the entire summation
		return -tf.tensordot(tf.convert_to_tensor(v_blank, dtype=tf.float32), summation_1, axes=3)

	#@tf.function
	def init_bin_distribution(self, data):
		"""
		Initializes self.bin_distribution to be the distribution for all bins over every image in the training set
		:param data, (training_set_size, h, w, 2) dimension matrix of all images in the training set
		"""
		#Initialized the distribution using all bins in all images
		batch_size = data.shape[0]
		r_size = data.shape[1]
		c_size = data.shape[2]
		labels = tf.reshape(data, [-1, 2])
		print("Finished reshape")
		bin_distributions = tf.map_fn(lambda x: self.calculate_bin_distribution_pixel(x[0], x[1]), labels, parallel_iterations=10)
		print("Finished map_fn")
		self.bin_distribution += tf.reduce_sum(tf.reshape(bin_distributions, [batch_size, r_size, c_size, self.num_classes]), axis=(0, 1, 2))
		print("Finished init_bin_distribution")

	#@tf.function
	def calculate_bin_distribution(self, labels):
		"""
		Converts the input image from a,b values to a bin distribution using calculate_bin_distribution_pixel
		:param labels, (batch_size, h, w, 2) dimension matrix of a single image's ab values
		:return: (batch_size, h, w, num_bins) dimension matrix of a single image's bin distribution (each pixel should
		only have five bins with probability values)
		"""
		print("Started calculate_bin_distribution")
		batch_size = labels.shape[0]
		r_size = labels.shape[1]
		c_size = labels.shape[2]
		labels = tf.reshape(labels, [-1, 2])
		#Form the bun distribution for the given pixels
		bin_distributions = tf.map_fn(lambda x: self.calculate_bin_distribution_pixel(x[0], x[1]), labels, parallel_iterations=10)
		#Reshape for normal shape
		bin_distributions = tf.reshape(bin_distributions, [batch_size, r_size, c_size, self.num_classes])
		print("Finished calculate_bin_distribution")
		return bin_distributions

	def calculate_bin_distribution_pixel(self, a, b):
		"""
		Converts the input image from a,b values to a bin distribution by calculating five nearest neighbors using a
		bounding box
		:param a, color component of a single pixel
		:param b, color component of a single pixel
		:return: (num_bins) dimension bin distribution for a single pixel(should only have five bins with probability
		values)
		"""
		bin_distribution = np.zeros(shape=[self.num_classes], dtype=np.float32)

		#Get the values of all regions in the boxed off space
		top_left = (a - (self.a_class_size / 2 + self.expansion_size), b + (self.b_class_size / 2 + self.expansion_size))
		top_right = (a + (self.a_class_size / 2 + self.expansion_size), b + (self.b_class_size / 2 + self.expansion_size))
		bot_left = (a - (self.a_class_size / 2 + self.expansion_size),b - (self.b_class_size / 2 + self.expansion_size))
		bot_right = (a + (self.a_class_size / 2 + self.expansion_size), b - (self.b_class_size / 2 + self.expansion_size))

		#Calculate the bins
		center_bin = self.ab_to_bin(a, b)
		top_left_bin = self.ab_to_bin(top_left[0], top_left[1])
		top_right_bin = self.ab_to_bin(top_right[0], top_right[1])
		bot_left_bin = self.ab_to_bin(bot_left[0], bot_left[1])
		bot_right_bin = self.ab_to_bin(bot_right[0], bot_right[1])

		#Retrieve the ab values for those bins
		top_left_center = self.bin_to_ab(top_left_bin)
		top_right_center = self.bin_to_ab(top_right_bin)
		bot_left_center = self.bin_to_ab(bot_left_bin)
		bot_right_center = self.bin_to_ab(bot_right_bin)

		#Calculate the distance to the ab centers from the original point
		top_left_dist = tf.norm(tf.subtract(top_left, top_left_center))
		top_right_dist = tf.norm(tf.subtract(top_right, top_right_center))
		bot_left_dist = tf.norm(tf.subtract(bot_left, bot_left_center))
		bot_right_dist = tf.norm(tf.subtract(bot_right, bot_right_center))

		dist = tfp.distributions.Normal(loc=0, scale=self.bin_distance_stddev)

		#Weight based on these distances
		top_left_prob = dist.prob(top_left_dist)
		top_right_prob = dist.prob(top_right_dist)
		bot_left_prob = dist.prob(bot_left_dist)
		bot_right_prob = dist.prob(bot_right_dist)
		center_prob = dist.prob(0)

		#Check for edge cases in the bins
		if self.check_in_ab_range(tf.convert_to_tensor(top_left[0]), tf.convert_to_tensor(top_left[1])):
			bin_distribution[top_left_bin] += top_left_prob

		if self.check_in_ab_range(top_right[0], top_right[1]):
			bin_distribution[top_right_bin] += top_right_prob

		if self.check_in_ab_range(bot_left[0], bot_left[1]):
			bin_distribution[bot_left_bin] += bot_left_prob

		if self.check_in_ab_range(bot_right[0], bot_right[1]):
			bin_distribution[bot_right_bin] += bot_right_prob

		bin_distribution[center_bin] += center_prob

		return bin_distribution

	def check_in_ab_range(self, a, b):
		"""
		Checks if an ab value is in range
		:param a, b: a,b value passed in
		:return: boolean representing whether it's in range
		"""
		return tf.math.less(self.a_min, a) and tf.math.less(a, self.a_max) and tf.math.less(self.b_min, b) and tf.math.less(b, self.b_max)

	def call(self, inputs):
		"""
		Runs a forward pass on an input batch of images.
		:param inputs: images, shape of (batch_size, w, h, 1); during training, the shape is (batch_size, 32, 32, 3)
		:return: logits - a matrix of shape (batch_size, w, h, 400); during training, it would be (batch_size, 2)
		"""
		return self.model(inputs)

	def accuracy(self, logits, labels):
		"""
		Calculates the model's prediction accuracy by comparing
		logits over bins to correct labels
		:param logits: a matrix of size (num_images, width, height, num_bins); This is the output of a call
		:param labels: matrix of size (num_images, width, height, 2) contains labels with just ab channel
		
		:return: the accuracy of the model as a Tensor
		"""
		# represents how far predicted ab values can be from labels to be counted as correct
		threshold = 5
		# get ab values for predicted images from bins
		final_images = self.h_function(logits)
		# get difference between ab values of predicted images and labels and then take the abs value
		diff = labels - final_images
		abs = tf.abs(diff)
		# if abs value is below a certain threshold count them as true
		correct_predictions = abs < threshold
		return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, manager, epoch, train_inputs, train_labels):
	"""
	Trains the model on all of the inputs and labels for one epoch.
	:param manager: to be used to save checkpoints
	:param epoch: just used to print the epoch number
	:param model: the initialized model to use for the forward pass and backward pass
	:param train_inputs: train inputs (all inputs to use for training),
	shape (num_inputs, width, height, 1) These are the black and white inputs
	:param train_labels: train labels (all labels to use for training),
	shape (num_images, width, height, 2) images with ab channels only
	:return: None
	"""
	# randomly shuffle images
	num_examples = train_inputs.shape[0]
	indices = range(num_examples)
	indices = tf.random.shuffle(indices)
	tf.gather(train_inputs, indices)
	tf.gather(train_labels, indices)
	train_inputs = tf.image.random_flip_left_right(train_inputs)

	# batch training
	batch = 0
	batch_start = 0
	while (batch_start + args.batch_size) < len(train_inputs):
		batch += 1
		batch_end = batch_start + args.batch_size
		if batch_end > len(train_inputs):
			batch_end = len(train_inputs)
		with tf.GradientTape() as tape:
			predictions = model.call(tf.cast(train_inputs[batch_start:batch_end, :, :, :], tf.float32))
			loss = model.loss(predictions, train_labels[batch_start:batch_end, :, :, :])
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
		# print information at every batch
		print("Epoch: {}/{} Batch: {}/{} Loss: {} Accuracy: {}".format(epoch + 1, args.num_epochs, batch,
																	   len(train_inputs) / args.batch_size, loss,
																	   model.accuracy(model.call(
																		   train_inputs[batch_start:batch_end, :, :,
																		   :]), train_labels[batch_start:batch_end,
																				:, :, 1:3])))
		batch_start += args.batch_size
		# currently we save checkpoints after every batch
		if batch % 1 == 0:
			manager.save()
			print("Saved Batch!")

	return None


def test(model, test_inputs, test_labels):
	"""
	Tests the model on the test inputs and labels. You should NOT randomly 
	flip images or do any extra preprocessing.
	:param test_inputs: test data (all images to be tested), 
	shape (num_inputs, width, height, 1) these are black and white images
	:param test_labels: test labels (all corresponding labels),
	shape (num_inputs, width, height, 2) just has ab channels
	:return: test accuracy
	"""
	test_logits = model.call(test_inputs)
	return model.accuracy(test_logits, test_labels)

def visualize_images(bw_images, color_images, predictions):
	"""
	:param bw_images: train_inputs (num_images, 32, 32, 1) just has L channel
	:param color_images: saved actual images (num_images, 32, 32, 3) has all Lab channels
	:param predictions: predicted images (num_images, 32, 32, 2) has the a and b channels
	:return: nothing, but saves jpg of output or displays it if --display TRUE is given
	"""
	num_images = bw_images.shape[0]

	fig, axs = plt.subplots(nrows=3, ncols=num_images)
	fig.suptitle("Images\n ")
	# we must reformat the predicted black and white images
	reformatted = np.zeros([bw_images.shape[0], bw_images.shape[1], bw_images.shape[2], 3])
	reformatted_predictions = np.zeros([bw_images.shape[0], bw_images.shape[1], bw_images.shape[2], 3])
	for i in range(bw_images.shape[0]):
		for w in range(bw_images.shape[1]):
			for h in range(bw_images.shape[2]):
				# makes black and white image now have 3 channels (where a and b are 0)
				reformatted[i, w, h, 0] = bw_images[i, w, h, 0]
				# reformat predictions to add back L channel
				reformatted_predictions[i, w, h, 0] = bw_images[i, w, h, 0]
				reformatted_predictions[i, w, h, 1] = predictions[i, w, h, 0]
				reformatted_predictions[i, w, h, 2] = predictions[i, w, h, 1]
	# This part is what actually displays the images
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
	if args.display:
		plt.show()
	else:
		# saves file as output.jpg
		plt.savefig('output.jpg', bbox_inches='tight')


def main():
	"""
	Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
	test your model for a number of epochs. We recommend that you train for
	10 epochs and at most 25 epochs. For CS2470 students, you must train within 10 epochs.
	You should receive a final accuracy on the testing examples for cat and dog of >=70%.
	:return: None
	"""

	model = Colorizer()
	checkpoint_dir = './checkpoints'
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(model=model)
	manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

	if args.mode == 'test' or args.skip_bin_init:
		# restores the latest checkpoint using from the manager if we are testing
		checkpoint.restore(manager.latest_checkpoint)

	# load data
	training_inputs, all_train_channels = get_data('CIFAR_data_compressed/train')
	test_inputs, all_test_channels = get_data('CIFAR_data_compressed/test')
	print("Finished importing data")

	# save full 3 channel images for visualizer and convert labels to just have ab channels
	visualizer_images = all_test_channels
	training_labels = all_train_channels[:, :, :, 1:]
	test_labels = all_test_channels[:, :, :, 1:]

	try:
		# specify an invalid GPU device
		with tf.device("/device:" + args.device):
			if args.mode == 'train':
				if not args.skip_bin_init:
					# getting bin distribution
					batch_start = 0
					batch = 0
					# batch the bin distribution initialization (this isn't training, just initializing our bin_distribution
					# array on the dataset)
					while (batch_start + args.bin_init_batch_size) < args.bin_init_images:
						batch += 1
						batch_end = batch_start + args.bin_init_batch_size
						if batch_end > len(training_inputs):
							batch_end = len(training_inputs)
						# call function to init the bin distribution
						model.init_bin_distribution(training_labels[batch_start:batch_end, :, :, :])
						print("Initializing Distribution Batch {}/{}".format(batch, args.bin_init_images/args.bin_init_batch_size))
						batch_start += args.bin_init_batch_size
					model.init_w()
					# save a checkpoint after this finishes
					manager.save()
					print("Saved Initializer")
				# actual training
				# loop through epochs
				for e in range(args.num_epochs):
					train(model, manager, e, training_inputs, training_labels)
			# Run this if testing
			if args.mode == 'test':
				if args.full_test:
					print("Testing!")
					# We batch testing
					batch_start = 0
					total = 0
					while (batch_start + args.batch_size) < len(test_inputs):
						batch_end = batch_start + args.batch_size
						if batch_end > len(training_inputs):
							batch_end = len(training_inputs)
						total += (batch_end - batch_start) * test(model, test_inputs[batch_start:batch_end, :, :, :],
									 test_labels[batch_start:batch_end, :, :, 1:3])
						batch_start += args.batch_size

					print("Final Accuracy: {}".format(total / len(test_inputs)))
				# If we aren't running a full test (which we don't do by default) just test on 5 images and visualize them
				if not args.full_test:
					print("Final Accuracy: {}".format(test(model, test_inputs[0:5, :, :], training_labels[0:5, :, :, :])))
				predictions = model.call(test_inputs[0:5, :, :])
				# visualizer outputs .jpg or displays them, depending on what is specified with --display
				visualize_images(test_inputs[0:5, :, :], visualizer_images[0:5, :, :, :], model.h_function(predictions))
	except RuntimeError as e:
		print(e)

	return


if __name__ == '__main__':
	main()

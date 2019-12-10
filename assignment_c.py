from __future__ import absolute_import
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from preprocess_c import get_data
from skimage import color

import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Activation
from tensorflow.keras.initializers import RandomNormal, TruncatedNormal
import numpy as np
import random
import argparse
from scipy.stats import norm
from scipy.ndimage import gaussian_filter


gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)

parser = argparse.ArgumentParser(description='Colorizer')

parser.add_argument('--mode', type=str, default='test', help='Can be "train" or "test"')
parser.add_argument('--device', type=str, default='GPU:0' if gpu_available else 'CPU:0',
                    help='specific the device of computation eg. CPU:0, GPU:0, GPU:1, GPU:2, ... ')
parser.add_argument('--batch-size', type=int, default=64,
                    help='Sizes of image batches fed through the network')
parser.add_argument('--num-epochs', type=int, default=10,
                    help='Number of passes through the training data to make before stopping')
parser.add_argument('--display', type=bool, default=False,
                    help='False saves file, True displays')
parser.add_argument('--full-test', type=bool, default=False,
                    help='True gets final accuracy')

args = parser.parse_args()

if not args.display:
	matplotlib.use('Agg')

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

		# other constants
		self.bin_to_ab_arr = np.zeros(shape=(self.num_a_partitions * self.num_b_partitions, 2), dtype=np.float32)
		self.expansion_size = 0.0001
		self.stdev = .04
		self.bin_distribution = np.zeros(shape=self.num_classes)
		self.bin_distance_stddev = 5
		self.gaussian_filter_stddev = 5
		self.lam = .5
		self.w = np.zeros(shape=self.num_classes)
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
		a_index = (a - self.a_min) / self.a_range
		b_index = (b - self.b_min) / self.b_range
		bin_num = int(a_index * self.num_a_partitions) * self.num_b_partitions + int(b_index * self.num_b_partitions)
		return bin_num

	def bin_to_ab(self, bin_id):
		"""
		Calculates a, b values for a given bin id
		:param bin_val: bin number (which is 0-indexed)
		:return: corresponding ab value as a tuple (a, b)
		"""
		a_index = bin_id / self.num_a_partitions
		b_index = bin_id % self.num_a_partitions
		a = (a_index + 0.5) / self.num_a_partitions * self.a_range + self.a_min
		b = (b_index + 0.5) / self.num_b_partitions * self.b_range + self.b_min
		return a, b

	def init_bin_to_ab_array(self):
		"""
		Fill in the values for the bin_to_ab array
		"""
		for i in range(self.bin_to_ab_arr.shape[0]):
			a, b = self.bin_to_ab(i)
			self.bin_to_ab_arr[i][0] = a
			self.bin_to_ab_arr[i][1] = b

	def init_w(self):
		"""
		Initializes self.w field based on the formula from the paper (page 6, equation 4)
		"""
		w = np.reciprocal(self.bin_distribution * (1 - self.lam) + self.lam / self.num_classes)
		unbalanced_expectation = np.dot(self.bin_distribution, w)
		self.w = gaussian_filter(w / unbalanced_expectation, sigma=self.gaussian_filter_stddev)

	def v(self, z):
		"""
		Weights for less common bins (page 6, equation 3)
		:param z, (batch_size, h, w, q) dimension, bin distribution for all images in a batch
		:return: weight to multiply by for the most likely bin
		"""
		pos = tf.math.argmax(z)
		return self.w[pos]

	def loss(self, predictions, labels):
		"""
		Calculates loss between probability distribution of bins for real and predicted image (page 5, equation 2)
		:param labels, (batch_size, h, w, 2) dimension matrix of real image's ab values
		:param predictions, (batch_size, h, w, q) dimension matrix of predicted
		:return: loss
		"""
		z = self.calculate_bin_distribution(labels)
		num_images = predictions.shape[0]
		h = predictions.shape[1]
		w = predictions.shape[2]
		v_blank = np.zeros(shape=(predictions.shape[:3]))
		summation_1 = tf.keras.losses.categorical_crossentropy(z, predictions)
			# tf.math.reduce_sum(tf.multiply(z, tf.math.log(predictions)), axis=3)




		for x in range(num_images):
			for i in range(h):
				for j in range(w):
					v_blank[x,i,j] = self.v(z[x,i,j,:])
		return -tf.tensordot(tf.convert_to_tensor(v_blank, dtype=tf.float32), summation_1, axes=3)

	def init_bin_distribution(self, data):
		"""
		Initializes self.bin_distribution to be the distribution for all bins over every image in the training set
		:param data, (training_set_size, h, w, 2) dimension matrix of all images in the training set
		"""
		for i in range(data.shape[0]):
			for r in range(data.shape[1]):
				for c in range(data.shape[2]):
					self.bin_distribution += self.calculate_bin_distribution_pixel(data[i][r][c][0], data[i][r][c][1])

	def calculate_bin_distribution(self, labels):
		"""
		Converts the input image from a,b values to a bin distribution using calculate_bin_distribution_pixel
		:param labels, (batch_size, h, w, 2) dimension matrix of a single image's ab values
		:return: (batch_size, h, w, num_bins) dimension matrix of a single image's bin distribution (each pixel should
		only have five bins with probability values)
		"""
		batch_size = labels.shape[0]
		r_size = labels.shape[1]
		c_size = labels.shape[2]
		labels = tf.reshape(labels, [-1, 2])
		bin_distributions = tf.map_fn(lambda x: self.calculate_bin_distribution_pixel(x[0], x[1]), labels)
		bin_distributions = tf.reshape(bin_distributions, [batch_size, r_size, c_size, self.num_classes])
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
		bin_distribution = np.zeros(shape=self.num_classes, dtype=np.float32)

		top_left = (a - (self.a_class_size / 2 + self.expansion_size), b + (self.b_class_size / 2 + self.expansion_size))
		top_right = (a + (self.a_class_size / 2 + self.expansion_size), b + (self.b_class_size / 2 + self.expansion_size))
		bot_left = (a - (self.a_class_size / 2 + self.expansion_size),b - (self.b_class_size / 2 + self.expansion_size))
		bot_right = (a + (self.a_class_size / 2 + self.expansion_size), b - (self.b_class_size / 2 + self.expansion_size))

		center_bin = self.ab_to_bin(a, b)
		top_left_bin = self.ab_to_bin(top_left[0], top_left[1])
		top_right_bin = self.ab_to_bin(top_right[0], top_right[1])
		bot_left_bin = self.ab_to_bin(bot_left[0], bot_left[1])
		bot_right_bin = self.ab_to_bin(bot_right[0], bot_right[1])

		top_left_center = self.bin_to_ab(top_left_bin)
		top_right_center = self.bin_to_ab(top_right_bin)
		bot_left_center = self.bin_to_ab(bot_left_bin)
		bot_right_center = self.bin_to_ab(bot_right_bin)

		top_left_dist = np.linalg.norm(tuple(np.subtract(top_left, top_left_center)))
		top_right_dist = np.linalg.norm(tuple(np.subtract(top_right, top_right_center)))
		bot_left_dist = np.linalg.norm(tuple(np.subtract(bot_left, bot_left_center)))
		bot_right_dist = np.linalg.norm(tuple(np.subtract(bot_right, bot_right_center)))

		top_left_prob = norm.pdf(top_left_dist, loc=0, scale=self.bin_distance_stddev)
		top_right_prob = norm.pdf(top_right_dist, loc=0, scale=self.bin_distance_stddev)
		bot_left_prob = norm.pdf(bot_left_dist, loc=0, scale=self.bin_distance_stddev)
		bot_right_prob = norm.pdf(bot_right_dist, loc=0, scale=self.bin_distance_stddev)
		center_prob = norm.pdf(0, loc=0, scale=self.bin_distance_stddev)

		if self.check_in_ab_range(top_left[0], top_left[1]):
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
		return self.a_min < a < self.a_max and self.b_min < b < self.b_max

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
	Trains the model on all of the inputs and labels for one epoch.
	:param model: the initialized model to use for the forward pass and backward pass
	:param train_inputs: train inputs (all inputs to use for training),
	shape (num_inputs, width, height, num_channels)
	:param train_labels: train labels (all labels to use for training),
	shape (num_labels, num_classes)
	:return: None
	"""
	num_examples = train_inputs.shape[0]
	indices = range(num_examples)
	indices = tf.random.shuffle(indices)
	tf.gather(train_inputs, indices)
	tf.gather(train_labels, indices)
	train_inputs = tf.image.random_flip_left_right(train_inputs)
	i = 0
	min_index = i * args.batch_size
	while min_index < num_examples:
		max_index = min_index + args.batch_size
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
		min_index = i * args.batch_size
	loss = model.loss(predictions, labels)
	return loss


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
	test_logits = model.call(test_inputs)
	return model.accuracy(test_logits, test_labels)

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
				reformatted_predictions[i, w, h, 1] = predictions[i, w, h, 0]
				reformatted_predictions[i, w, h, 2] = predictions[i, w, h, 1]
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

	if args.mode == 'test':
		# restores the latest checkpoint using from the manager
		checkpoint.restore(manager.latest_checkpoint)

	training_inputs, all_train_channels = get_data('CIFAR_data_compressed/train')
	# test_inputs, all_test_channels = get_data('CIFAR_data_compressed/test')
	print("Finished importing data")

	training_labels = all_train_channels[:, :, :, 1:]
	# test_labels = all_test_channels[:, :, :, 1:]

	try:
		# specify an invalid GPU device
		with tf.device("/device:" + args.device):
			if args.mode == 'train':
				# getting bin distribution
				model.init_bin_distribution(training_labels[:1, :, :, :])
				model.init_w()
				# actual training
				for e in range(args.num_epochs):
					batch = 0
					batch_start = 0
					while (batch_start + args.batch_size) < len(training_inputs):
						batch += 1
						batch_end = batch_start + args.batch_size
						if batch_end > len(training_inputs):
							batch_end = len(training_inputs)
						loss = train(model, training_inputs[batch_start:batch_end, :, :, :], training_labels[batch_start:batch_end, :, :, :])
						print("Epoch: {}/{} Batch: {}/{} Loss: {} Accuracy: {}".format(e + 1, args.num_epochs, batch, len(training_inputs)/args.batch_size, loss, model.accuracy(model.call(training_inputs[batch_start:batch_end, :, :, :]), training_labels[batch_start:batch_end, :, :, 1:3])))
						batch_start += args.batch_size
						if batch % 50 == 0:
							manager.save()
							print("Saved Batch!")
			if args.mode == 'test':
				if args.full_test:
					print("Testing!")

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

				predictions = model.call(test_inputs[0:5, :, :])

				visualize_images(test_inputs[0:5, :, :], test_labels[0:5, :, :], predictions)
	except RuntimeError as e:
		print(e)

	return


if __name__ == '__main__':
	main()

import pickle
import numpy as np
import tensorflow as tf
#scimg = __import__('scikit-image')
#color = scimg.color()
from skimage import color

def unpickle(file):
	"""
	CIFAR data contains the files data_batch_1, data_batch_2, ..., 
	as well as test_batch. We have combined all train batches into one
	batch for you. Each of these files is a Python "pickled" 
	object produced with cPickle. The code below will open up a 
	"pickled" object (each file) and return a dictionary.

	NOTE: DO NOT EDIT

	:param file: the file to unpickle
	:return: dictionary of unpickled data
	"""
	print(file)
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def get_data(file_path):
    unpickled_file = unpickle(file_path)
    inputs = unpickled_file[b'data']

    # inputs are a set of 32 x 32 images from the CIFAR-10 datasest. Each has 3 channels of RGB.
    inputs = np.reshape(inputs, (-1, 3, 32, 32))
    inputs = np.transpose(inputs, (0, 2, 3, 1))
    inputs = inputs / float(255)
    inputs = inputs.astype('float32')
    # use rgb2lab conversion to convert each image to the Lab space we use for colorizing (instead of RGB)
    for i in range(inputs.shape[0]):
        inputs[i] = color.rgb2lab(inputs[i])
    # we will use the current inputs as our labels and then remove the color from our inputs
    labels = np.copy(inputs)
    # This removes all channels from the inputs except the L channel, so they are now black and white
    inputs = inputs[:, :, :, 0]
    # converts inputs from shape (num_images, 32,32) to (num_images, 32, 32, 1)
    inputs = np.expand_dims(inputs, axis=3)
    return inputs, labels

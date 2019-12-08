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

    # inputs = inputs[indices]
    inputs = np.reshape(inputs, (-1, 3, 32, 32))
    inputs = np.transpose(inputs, (0, 2, 3, 1))
    inputs = inputs / float(255)
    inputs = inputs.astype('float32')
    for i in range(inputs.shape[0]):
        inputs[i] = color.rgb2lab(inputs[i])
    labels = np.copy(inputs)
    inputs = inputs[:, :, :, 0]
    inputs = np.expand_dims(inputs, axis=3) #converts from (num_images, 32,32) to (num_images, 32, 32, 1)
    return inputs, labels

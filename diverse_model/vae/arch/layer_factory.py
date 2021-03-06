import tensorflow as tf
import numpy as np
from tensorflow.python.framework import tensor_shape

class layer_factory:

	def __init__(self):
		pass

	def weight_variable(self, name, shape=None, mean=0., stddev=.001, gain=np.sqrt(2)):
		if(shape == None):
			return tf.compat.v1.get_variable(name)
#		#Adaptive initialize based on variable shape
#		if(len(shape) == 4):
#			stddev = (1.0 * gain) / np.sqrt(shape[0] * shape[1] * shape[3])
#		else:
#			stddev = (1.0 * gain) / np.sqrt(shape[0])
		return tf.compat.v1.get_variable(name, shape=shape, initializer=tf.compat.v1.random_normal_initializer(mean=mean, stddev=stddev))
	
	def bias_variable(self, name, shape=None, constval=.001):
		if(shape == None):
			return tf.compat.v1.get_variable(name)
		return tf.compat.v1.get_variable(name, shape=shape, initializer=tf.compat.v1.constant_initializer(constval))

	def conv2d(self, x, W, stride=1, padding='SAME'):
		return tf.nn.conv2d(input=x, filters=W, strides=[1, stride, stride, 1], padding=padding)
	
	def lrelu(self, x, leak=.2):
		return tf.maximum(x, leak*x)

	def batch_norm_aiuiuc_wrapper(self, x, train_phase, name, reuse_vars):
		output = tf.compat.v1.layers.batch_normalization(x, momentum=.99,training=train_phase,scale=True, epsilon=1e-4,name=name,reuse=reuse_vars)
	# 	output = tf.contrib.layers.batch_norm(x, \
    #   decay=.99, \
    #   is_training=train_phase, \
    #   scale=True,  \
    #   epsilon=1e-4, \
    #   updates_collections=None,\
    #   scope=name,\
    #   reuse=reuse_vars)
		return output

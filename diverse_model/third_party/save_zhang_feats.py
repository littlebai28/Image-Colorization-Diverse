import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"   
import sys

import tensorflow as tf

import torch 
import numpy as np
#import caffe
import skimage.color as color
import skimage.io
import scipy.ndimage.interpolation as sni
import cv2
import matplotlib.pyplot as plt
from torchvision.models.feature_extraction import create_feature_extractor
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

from third_party.colorizers import *

def save_zhang_feats(img_fns, ext='JPEG'):

	# gpu_id = 0
	# caffe.set_mode_gpu()
	# caffe.set_device(gpu_id)
	
	#third_party/colorization/models/colorization_deploy_v1.prototxt --caffemodel third_party/colorization/models/colorization_release_v1.caffemodel --data-output-path case_tf.npy --code-output-path case_tf.py
	#mmtocode -f tensorflow --IRModelPath caffe_IR.pb --IRWeightPath caffe_IR.npy --dstModelPath tf_diverse.py
	#mmtomodel -f tensorflow -in tf_diverse.py -iw caffe_IR.npy -o tf_diverse_w --dump_tag SERVING

	# net = caffe.Net('third_party/colorization/models/colorization_deploy_v1.prototxt', \
    # 'third_party/colorization/models/colorization_release_v1.caffemodel', caffe.TEST)
	# loaded = tf.saved_model.load("tf_diverse_v3")
	# net = tf.keras.models.load_model('saved_model')
	colorizer_eccv16 = eccv16(pretrained=True).eval()
	print(colorizer_eccv16)
	return_layers = {
    'model7.4': 'conv7_3',
	}
	mid_getter = MidGetter(eccv16(pretrained=True), return_layers=return_layers, keep_output=True)
	# activation = {}
	# def get_activation(name):
	# 	def hook(model, input, output):
	# 		activation[name] = output.detach()
	# 	return hook

	# eccv16().model7.register_forward_hook(get_activation('conv7_3'))
	
	# print(net.layers)
	# class LayerFromSavedModel(tf.keras.layers.Layer):
	# 	def __init__(self):
	# 		super(LayerFromSavedModel, self).__init__()
	# 		self.vars = loaded.variables
	# 	def call(self, inputs):
	# 		print(inputs[0,:,:,0])
	# 		return loaded.signatures['serving_default'](inputs)

	# input = tf.keras.Input(dtype = tf.float32, shape = (224, 224, 1))
	# model = tf.keras.Model(input, LayerFromSavedModel()(input))
	# print(model.summary())
	# print(model.layers[1])
	# model.save('saved_model')
	# print(net.signatures)
	# print(net.layers[0].output_shape)
	(H_in,W_in) = (224,224)#net.get_layer('data_l').output_shape[2:] # get input shape
	#(H_out,W_out) = net.blobs['class8_ab'].data.shape[2:] # get output shape
	#net.blobs['Trecip'].data[...] = 6/np.log(10) # 1/T, set annealing temperature

	feats_fns = []
	for img_fn_i, img_fn in enumerate(img_fns):

		# load the original image
		# from PIL import Image
		img_rgb = tf.keras.utils.load_img(img_fn)
		# img = np.array(Image.open(img_fn).convert('L').resize((H_in,W_in), Image.ANTIALIAS))
		#print(img.shape)
		img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
		img_l = img_lab[:,:,0] # pull out L channel
		#(H_orig,W_orig) = img_rgb.shape[:2] # original image size

		# create grayscale version of image (just for displaying)
		img_lab_bw = img_lab.copy()
		img_lab_bw[:,:,1:] = 0
		img_rgb_bw = color.lab2rgb(img_lab_bw)

		# resize image to network input size
		img_rs = tf.image.resize(img_rgb,(H_in,W_in)) # resize image to network input size
		img_lab_rs = color.rgb2lab(img_rs)
		img_l_rs = img_lab_rs[:,:,0]
		img = img_l_rs.reshape((1,224,224,1))
		#print(img_l_rs.shape)
		# print(loaded.signatures['serving_default'].structured_outputs)
		# x= tf.keras.preprocessing.image.load_img(img_fn, target_size=[224, 224])
		# x = tf.convert_to_tensor(x,dytpe =tf.float32)
		# plt.imshow(tf.keras.preprocessing.image.load_img(img_fn, target_size=[224, 224]))
		# plt.axis('off')
		# plt.show()
		#print(img[:,:,0])

		img = load_img(img_fn)
		
		(tens_l_orig, tens_l_rs) = preprocess_img(img , HW=(256,256))
		img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
		out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs-50).cpu())
		mid_outputs, model_output = mid_getter(tens_l_rs)
		inter = mid_outputs['conv7_3'][0].detach().numpy()[:,4:,4:]
		np.nan_to_num(inter)	
		#print(mid_outputs['conv7_3'][0])
		#print(out_img_eccv16)
		# plt.figure(figsize=(12,8))
		# plt.subplot(2,2,1)
		# plt.imshow(mid_outputs['conv7_3'][0].detach().numpy())
		# plt.title('Original')
		# plt.axis('off')

		# plt.subplot(2,2,2)
		# plt.imshow(model_output)
		# plt.title('Input')
		# plt.axis('off')

		# plt.subplot(2,2,3)
		# plt.imshow(out_img_eccv16)
		# plt.title('Output (ECCV 16)')
		# plt.axis('off')

		# plt.show()

	
		# colorizer outputs 256x256 ab map
		# resize and concatenate to original L channel
		#out = loaded.signatures['serving_default'](tf.constant(img,dtype = tf.float32))
		#model.predict(img)

		# net.blobs['data_l'].data[0,0,:,:] = img_l_rs-50 # subtract 50 for mean-centering
		# net.forward() # run network
		
		#print(activation['conv7_3'])
		npz_fn = img_fn.replace(ext, 'npz')
		print(np.ptp(inter))
		print(np.ptp(out_img_eccv16))
		# print(min(mid_outputs['conv7_3'][0].detach().numpy()[:,2:-2,2:-2]).all())
		# print(max(out_img_eccv16).all())
		# print(min(out_img_eccv16).all())
		inter[inter == float("Inf")] = 0
		print(np.count_nonzero(np.isnan(inter)))
		print(inter.shape)
		np.savez_compressed(npz_fn,inter)#net.blobs['conv7_3'].data
		feats_fns.append(npz_fn)

	return feats_fns

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"   
# import sys


# import numpy as np
# import caffemodel2pytorch as caffe
# import skimage.color as color
# import skimage.io
# import scipy.ndimage.interpolation as sni
# import cv2

# def save_zhang_feats(img_fns, ext='JPEG'):

# 	# gpu_id = 0
# 	# caffe.set_mode_gpu()
# 	# caffe.set_device(gpu_id)
# 	#/Users/muhuaxu/Desktop/SophomoreSpring/6.869/Image-Colorization-Diverse/diverse_model/third_party/colorization/models/colorization_deploy_v1.prototxt
# 	#/Users/muhuaxu/Desktop/SophomoreSpring/6.869/Image-Colorization-Diverse/diverse_model/third_party/colorization/models/colorization_release_v1.caffemodel 
# 	net = caffe.Net('third_party/colorization/models/colorization_deploy_v1.prototxt',caffe.TEST, weights = 'third_party/colorization/models/colorization_release_v1.caffemodel',caffe_proto = 'https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto')
	
# 	print(net.blobs['data_l'].numpy)
# 	(H_in,W_in) = net.blobs['data_l'].data.shape[2:] # get input shape
# 	(H_out,W_out) = net.blobs['class8_ab'].data.shape[2:] # get output shape
# 	net.blobs['Trecip'].data[...] = 6/np.log(10) # 1/T, set annealing temperature

# 	feats_fns = []
# 	for img_fn_i, img_fn in enumerate(img_fns):

# 		# load the original image
# 		img_rgb = caffe.io.load_image(img_fn)
# 		img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
# 		img_l = img_lab[:,:,0] # pull out L channel
# 		(H_orig,W_orig) = img_rgb.shape[:2] # original image size

# 		# create grayscale version of image (just for displaying)
# 		img_lab_bw = img_lab.copy()
# 		img_lab_bw[:,:,1:] = 0
# 		img_rgb_bw = color.lab2rgb(img_lab_bw)

# 		# resize image to network input size
# 		img_rs = caffe.io.resize_image(img_rgb,(H_in,W_in)) # resize image to network input size
# 		img_lab_rs = color.rgb2lab(img_rs)
# 		img_l_rs = img_lab_rs[:,:,0]

# 		net.blobs['data_l'].data[0,0,:,:] = img_l_rs-50 # subtract 50 for mean-centering
# 		net.forward() # run network

# 		npz_fn = img_fn.replace(ext, 'npz')
# 		np.savez_compressed(npz_fn, net.blobs['conv7_3'].data)
# 		feats_fns.append(npz_fn)

# 	return feats_fns

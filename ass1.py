import matplotlib.pyplot as plt
import numpy as np
import os
import tarfile
import urllib
from PIL import Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
import cPickle as pickle
import random

url = 'http://yaroslavvb.com/upload/notMNIST/'

def maybe_download(filename, expected_bytes):
	"""Download a file if no present"""
	if not os.path.exists(filename):
		filename, _ = urllib.urlretrieve(url + filename, filename)
	statinfo = os.stat(filename)
	if statinfo.st_size == expected_bytes:
		print "Found and verified", filename
	else:
		raise Exception('Faildes to verify ' + filename)
	return filename
	
train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
num_classes = 10

def extract(filename):
	if not os.path.exists(filename):
		tar = tarfile.open(filename)
		tar.extractall()
		tar.close()
	root = os.path.splitext(os.path.splitext(filename)[0])[0]
	data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root))]
	if len(data_folders) != num_classes:
		raise Exception(
			'Expected &d folders, one per class. Found %d instead.' % (
				num_classes, len(data_folders)))
	#print data_folders
	return data_folders
	
train_folders = extract(train_filename)
test_folders = extract(test_filename)

display_samples = 0
display_folders = random.sample(train_folders, display_samples/2)
display_folders += random.sample(test_folders, display_samples/2)
display_filenames = []

for folder in display_folders:
	files = [os.path.join(folder, d) for d in os.listdir(folder)]
	file = random.sample(files, 1)[0]
	display_filenames.append(file)
	img = Image.open(file)
	img.show()
	

	
image_size = 28
pixel_depth = 255.0

def load(data_folders, min_num_images, max_num_images):
	dataset = np.ndarray(
		shape=(max_num_images, image_size, image_size), dtype=np.float32)
	labels = np.ndarray(shape=(max_num_images), dtype=np.int32)
	label_index = 0
	image_index = 0
	for folder in data_folders:
		print folder
		for image in os.listdir(folder):
			if image_index >= max_num_images:
				raise Exception('More images than expected')
			image_file = os.path.join(folder, image)
			try:
				image_data = (ndimage.imread(image_file).astype(float) - 
					pixel_depth / 2) / pixel_depth
				if image_data.shape != (image_size, image_size):
					raise Exception('Unexpected shape')
				dataset[image_index, :, :] = image_data
				labels[image_index] = label_index
				image_index += 1
			except IOError as e:
				print 'Could not read:', image_file
		label_index += 1
	num_images = image_index
	dataset = dataset[0:num_images, :, :]
	labels = labels[0: num_images]
	if(num_images < min_num_images):
		raise Exception('Many fewer images that expected')
	print 'Full data set Tensor:', dataset.shape
	print 'Mean:', np.mean(dataset)
	print 'Standard Deviation:', np.std(dataset)
	print 'Labels:', labels.shape
	return dataset, labels.shape

train_dataset, train_labels = load(train_folders, 450000, 550000)
test_dataset, train_dataset = laod(test_folders, 18000, 20000)

	

	

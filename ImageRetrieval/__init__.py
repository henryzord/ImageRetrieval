__author__ = 'Henry'


import os
import cv2
from features import *
from utils import *
from matplotlib import pyplot as plt

CHANNELS = ('b', 'g', 'r')

def main():
	query_size = 10

	train_database = load_database('train')
	test_database = load_database('test')

	for image in test_database:
		plt.figure()
		plt.imshow(image)

	plt.show()


# ############ #
# Thomas' code #
# ############ #

# path = "database/image.orig/"
# pre_processing_path = "pre_processing/"
# pre_processing_file = "features.txt"
# test_set = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# images_test_set = []

# image_a = cv2.imread(path + '1.jpg')
# image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)
# # imageB = cv2.imread(path + '1.jpg', cv2.IMREAD_COLOR)
# # imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
# # print compareTwoImages(imageA, imageB)
# some_similarity = pre_processing(image_a, 50)
# some_similarity = sorted(some_similarity, key=get_key)
#
# for i in range(10):
# print some_similarity[i]
#
# # retrieveSimilarImages()


def pre_processing1(path, filename, test_set):
	result_file = path + filename


	if os.path.exists(result_file):
		print 'Pre-processed file already exists in \'%s\'' % result_file
		return

	some_file = open(result_file, 'w')
	for image in os.listdir(path):
		if int(image[:len(image) - 4]) not in test_set:
			img = cv2.imread(path + image, cv2.IMREAD_COLOR)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			b, g, r = mean_rgb(img)
			some_histogram = histogram(img)
			line = '%s,%s,%s,%s,%s\n' % (image, b, g, r, some_histogram)
			some_file.write(line)
	some_file.close()


def pre_processing(image_a, path, test_set, limit=50):
	some_similarity = []
	for imageFile in os.listdir(path):
		if int(imageFile[:len(imageFile) - 4]) not in test_set:
			print 'Processing image %s' % imageFile
			image_b = cv2.imread(path + imageFile)
			image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)
			value = utils.compare_two_images(image_a, image_b)
			some_similarity.append((imageFile, value))

			if len(some_similarity) == limit:
				break

	return some_similarity


def process():
	pass


def get_key(item):
	return item[1]


def load_database(mode='train'):
	"""
	Method based on Henry code. May be replaced with Thomas' method.

	WARNING: either way, the channels of the images are RGB, but OpenCV loads them as BGR.
		This method already does this.
	:param mode: either train or test.
	:return:
	"""
	if mode == 'train':
		n_classes = 10
		files_per_class = []
		for i in xrange(n_classes):
			i_images = []

			path_to_class = os.path.join('..', 'images', mode, 'classe' + str(i))
			i_files = os.listdir(path_to_class)
			for some_file in i_files:
				some_image = cv2.imread(os.path.join(path_to_class, some_file))
				# flips from BGR to RGB
				some_image = cv2.cvtColor(some_image, cv2.COLOR_BGR2RGB)
				i_images += [some_image]
			files_per_class += [i_images]
	elif mode == 'test':
		path = os.path.join('..', 'images', mode)
		image_names = os.listdir(path)
		images = []
		for some_file in image_names:
			some_image = cv2.imread(os.path.join(path, some_file))
			some_image = cv2.cvtColor(some_image, cv2.COLOR_BGR2RGB)
			images += [some_image]

		return images

	else:
		raise NameError('invalid database mode!')

	return files_per_class


def compare_histogram(a_hist, b_hist, method=cv2.HISTCMP_CHISQR):
	"""
	Compares two histograms, whether they represent greyscale images
		or colored ones.
	:param a_hist: image A histogram.
	:param b_hist: image B histogram.
	:param method: Comparison method. May be one of the following:

		CV_COMP_CORREL - Correlation

		CV_COMP_CHISQR - Chi-Square

		CV_COMP_INTERSECT - Intersection

		CV_COMP_BHATTACHARYYA - Bhattacharyya distance

		CV_COMP_HELLINGER - Synonym for CV_COMP_BHATTACHARYYA

		Please refer to OpenCV documentation for further details.
	:return: The result of the comparison between two histograms.
	"""
	if isinstance(a_hist, list) and isinstance(b_hist, list):
		diff = []
		for i, channel in enumerate(CHANNELS):
			diff += [cv2.compareHist(a_hist[i], b_hist[i], method=method)]
	else:
		diff = cv2.compareHist(a_hist, b_hist, method=cv2.HISTCMP_CHISQR)

	return np.mean(diff)


def plot_figure(img):
	plt.figure()
	plt.imshow(img)


def get_histogram(img, **kwargs):
	greyscale = False if 'greyscale' not in kwargs else kwargs['greyscale']

	if greyscale:
		raise NameError('not implemented yet!')
	else:
		hist = []
		for i, col in enumerate(CHANNELS):
			hist += [cv2.calcHist([img], [i], None, [256], [0, 256])]
		return hist


def plot_histogram(**kwargs):
	greyscale = False if 'greyscale' not in kwargs else kwargs['greyscale']
	img = None if 'img' not in kwargs else kwargs['img']
	hist = None if 'hist' not in kwargs else kwargs['hist']

	plt.figure()

	if img is not None:
		if greyscale:
			hist = cv2.calcHist([img], [0], None, [256], [0, 256])
		else:
			hist = list()
			for i, col in enumerate(CHANNELS):
				hist += [cv2.calcHist([img], [i], None, [256], [0, 256])]

	if isinstance(hist, list):
		for i, col in enumerate(CHANNELS):
			plt.plot(hist[i], color=col)
			plt.xlim([0, 256])
	else:
		plt.hist(img.ravel(), 256, [0, 256])


if __name__ == '__main__':
	main()

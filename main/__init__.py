__author__ = 'Henry'

from matplotlib import pyplot as plt
import cv2
import os
import numpy as np

CHANNELS = ('b', 'g', 'r')


def main():
	query_size = 10

	train_database = load_database('train')
	test_database = load_database('test')

	for image in test_database:
		plt.figure()
		plt.imshow(image)

	plt.show()


def load_database(mode='train'):
	if mode == 'train':
		n_classes = 10
		files_per_class = []
		for i in xrange(n_classes):
			i_images = []

			path_to_class = os.path.join('..', 'images', mode, 'classe' + str(i))
			i_files = os.listdir(path_to_class)
			for some_file in i_files:
				i_images += [cv2.imread(os.path.join(path_to_class, some_file))]
			files_per_class += [i_images]
	elif mode == 'test':
		path = os.path.join('..', 'images', mode)
		image_names = os.listdir(path)
		images = []
		for some_file in image_names:
			images += [cv2.imread(os.path.join(path, some_file))]

		return images

	else:
		raise NameError('invalid database mode!')

	return files_per_class


def compare_histograms(a_hist, b_hist, method=cv2.HISTCMP_CHISQR):
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

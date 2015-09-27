__author__ = 'Henry, Thomas'

import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.measure import structural_similarity as ssim

from ImageRetrieval import CHANNELS


def mean_rgb(image):
	"""
	:param image:
	:return:
	"""
	# TODO document me thomas!
	b, g, r, alpha = cv2.mean(image)
	return b, g, r


def mean_squared_error(image_a, image_b):
	"""

	:param image_a:
	:param image_b:
	:return:
	"""

	# TODO document me thomas!
	squared_error = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
	squared_error /= float(image_a.shape[0] * image_a.shape[1])
	return squared_error


def get_histogram(img, **kwargs):
	"""
	Given an (colored) image, gets its three histograms (if greyscale is False)
		or its unique greyscale histogram.
	:param img: An colored image.
	:param greyscale: optional - whether to return one or three histograms, based on
		the colors of the image.
	:return: One or three histograms, depending on the greyscale flag.
	"""
	greyscale = False if 'greyscale' not in kwargs else kwargs['greyscale']

	if greyscale:
		raise NameError('not implemented yet!')
	else:
		hist = []
	for i, col in enumerate(CHANNELS):
		hist += [cv2.calcHist([img], [i], None, [256], [0, 256])]

	return hist


def plot_histogram(**kwargs):
	"""
	Plots one histogram.
	:param img: optional - if provided, will automatically compute the histogram
		of an (colored) image and display it. If provided, greyscale must be also supplied.
	:param greyscale: optional - whether to plot one greyscale histogram or three histograms.
		Must be provided alongside img.
	:param hist: optional - may be either one greyscale or three-channeled histogram.
	"""
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


def histogram(image, normalize=True):
	"""
	Calculates the histogram of an (colored) image.
	:param image:
	:param normalize:
	:return:
	"""
	hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
	if normalize:
		cv2.normalize(hist, hist).flatten()
	return hist


def hausdorff_distance(a_image, b_image):
	"""
	Measures the shape similarity between two (RGB colored) images.
	:return: The Hausdorff Distance (https://en.wikipedia.org/wiki/Hausdorff_distance) between
		two images.
	"""
	a_gray = cv2.cvtColor(a_image, cv2.COLOR_RGB2GRAY)
	ret_a, thresh_a = cv2.threshold(a_gray, 127, 255, 0)
	img_a, contours_a, hierarchy_a = cv2.findContours(thresh_a, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


	b_gray = cv2.cvtColor(b_image, cv2.COLOR_RGB2GRAY)
	ret_b, thresh_b = cv2.threshold(b_gray, 127, 255, 0)
	img_b, contours_b, hierarchy_b = cv2.findContours(thresh_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	hd = cv2.createHausdorffDistanceExtractor()
	distance = hd.computeDistance(contours_a[0], contours_b[0])

	return distance


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


def structural_similarity(image_a, image_b):
	"""

	:param image_a:
	:param image_b:
	:return:
	"""
	# TODO document me thomas!
	im_a = np.copy(image_a[0:255,0:255])
	im_b = np.copy(image_b[0:255,0:255])
	return ssim(im_a, im_b)


def compare_images(image_a, image_b):
	"""

	:param image_a:
	:param image_b:
	:return:
	"""
	# TODO document me thomas!
	# ssim = utils.get_structural_similarity(image_a, image_b)
	# hist = utils.get_histogram_comparison(image_a, image_b)
	sq = mean_squared_error(image_a, image_b)
	return sq
__author__ = 'Henry, Thomas'

import os
from features import *


def main():
	query_size = 10

	# train_database = load_database('train')
	# test_database = load_database('test')

	a_image = cv2.imread('..\\images\\test\\1.jpg')
	a_image = cv2.cvtColor(a_image, cv2.COLOR_BGR2RGB)

	b_image = cv2.imread('..\\images\\test\\101.jpg')
	b_image = cv2.cvtColor(b_image, cv2.COLOR_BGR2RGB)

	c_image = cv2.imread('..\\images\\train\\classe0\\9.jpg')
	c_image = cv2.cvtColor(c_image, cv2.COLOR_BGR2RGB)

	a_hist = get_histogram(a_image)
	b_hist = get_histogram(b_image)
	c_hist = get_histogram(c_image)

	distances = dict()
	distances[hausdorff_distance(a_image, c_image)] = 'a'
	distances[hausdorff_distance(b_image, c_image)] = 'b'

	histograms = dict()
	histograms[compare_histogram(a_hist, c_hist)] = 'a'
	histograms[compare_histogram(b_hist, c_hist)] = 'b'

	# TODO fazer uma abertura e um fechamento!
	closest_distance = min(distances.keys())
	closest_histogram = min(histograms.keys())

	print 'distance:', distances[closest_distance], 'histogram:', histograms[closest_histogram]
	for img in [a_image, b_image, c_image]:
		plt.figure()
		plt.imshow(img)
	plt.show()


def draw_contours(colored_img):
	"""
	Given an (RGB colored) image, draw its contours.
	:param colored_img: An RGB colored image.
	"""
	gray_img = cv2.cvtColor(colored_img, cv2.COLOR_RGB2GRAY)
	ret, thresh = cv2.threshold(gray_img, 127, 255, 0)
	contours_img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# for drawing contours:
	colored_img = cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 3)
	plt.imshow(colored_img)


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
			value = compare_images(image_a, image_b)
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


def plot_figure(img):
	plt.figure()
	plt.imshow(img)


if __name__ == '__main__':
	main()

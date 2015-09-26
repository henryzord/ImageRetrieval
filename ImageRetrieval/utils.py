from skimage.measure import structural_similarity as ssim
from features import *
import numpy as np
import cv2


def compare_two_images(image_a, image_b):
    #ssim = utils.get_structural_similarity(image_a, image_b)
    #hist = utils.get_histogram_comparison(image_a, image_b)
    sq = mean_squared_error(image_a, image_b)
    return sq

def structural_similarity(image_a, image_b):
    im_a = np.copy(image_a[0:255,0:255])
    im_b = np.copy(image_b[0:255,0:255])
    return ssim(im_a, im_b)
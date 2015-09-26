import numpy as np
import cv2


def mean_rgb(image):
    b, g, r, alpha = cv2.mean(image)
    return b, g, r


def mean_squared_error(image_a, image_b):
    squared_error = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    squared_error /= float(image_a.shape[0] * image_a.shape[1])
    return squared_error


def histogram(image, normalize=True):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    if normalize:
        cv2.normalize(hist, hist).flatten()
    return hist
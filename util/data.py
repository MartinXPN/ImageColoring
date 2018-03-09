import numpy as np
from skimage.color import rgb2lab, lab2rgb


def network_prediction_to_rgb(prediction, inputs):
    lab = np.concatenate((inputs, prediction), axis=2)
    lab[:, :, :1] *= 100.
    lab[:, :, 1:] *= 128.
    return lab2rgb(lab)


def rgb_to_target_image(rgb_image):
    lab = rgb2lab(rgb_image / 255.)
    lab[:, :, :1] /= 100.
    lab[:, :, 1:] /= 128.
    return lab


def rgb_to_colorizer_input(rgb_image):
    lab = rgb2lab(rgb_image / 255.)
    res = lab[:, :, :1]
    res /= 100.
    return res

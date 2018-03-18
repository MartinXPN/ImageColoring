import numpy as np
from skimage.color import yuv2rgb, rgb2yuv


def network_prediction_to_rgb(prediction, inputs):
    yuv = np.concatenate((inputs, prediction), axis=2)
    yuv[:, :, 1:] /= 2.
    return yuv2rgb(yuv)


def rgb_to_target_image(rgb_image):
    yuv = rgb2yuv(rgb_image / 255.)
    yuv[:, :, 1:] *= 2.
    return yuv


def rgb_to_colorizer_input(rgb_image):
    yuv = rgb2yuv(rgb_image / 255.)
    res = yuv[:, :, :1]
    return res

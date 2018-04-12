import numpy as np
from skimage.color import yuv2rgb, rgb2yuv, lab2rgb, rgb2lab


class DataMapper(object):
    def network_prediction_to_rgb(self, prediction, inputs):
        raise NotImplementedError('You need to implement mapping from a colorizer network prediction to a valid'
                                  'rgb image')

    def rgb_to_target_image(self, rgb_image):
        raise NotImplementedError('You need to implement mapping from rgb to a valid target image:\n'
                                  'An image from target distribution that could be used by critic to classify it as an'
                                  'image from real data distribution')

    def rgb_to_colorizer_target(self, rgb_image):
        raise NotImplementedError('You need to implement mapping from rgb to a valid colorizer target image:\n'
                                  'An image that could be used to pre-train colorizer using it as a label')

    def rgb_to_colorizer_input(self, rgb_image):
        raise NotImplementedError('You need to implement mapping from rgb to a valid colorizer input')


class YUVMapper(DataMapper):
    def network_prediction_to_rgb(self, prediction, inputs):
        yuv = np.concatenate((inputs, prediction), axis=2)
        yuv[:, :, 1:] /= 2.
        return yuv2rgb(yuv)

    def rgb_to_target_image(self, rgb_image):
        yuv = rgb2yuv(rgb_image / 255.)
        yuv[:, :, 1:] *= 2.
        return yuv

    def rgb_to_colorizer_target(self, rgb_image):
        return self.rgb_to_target_image(rgb_image)[:, :, 1:]

    def rgb_to_colorizer_input(self, rgb_image):
        yuv = rgb2yuv(rgb_image / 255.)
        res = yuv[:, :, :1]
        return res


class LabMapper(DataMapper):
    def network_prediction_to_rgb(self, prediction, inputs):
        lab = np.concatenate((inputs, prediction), axis=2)
        lab[:, :, :1] *= 100.
        lab[:, :, 1:] *= 128.
        return lab2rgb(lab)

    def rgb_to_target_image(self, rgb_image):
        lab = rgb2lab(rgb_image / 255.)
        lab[:, :, :1] /= 100.
        lab[:, :, 1:] /= 128.
        return lab

    def rgb_to_colorizer_target(self, rgb_image):
        return self.rgb_to_target_image(rgb_image)[:, :, 1:]

    def rgb_to_colorizer_input(self, rgb_image):
        lab = rgb2lab(rgb_image / 255.)
        res = lab[:, :, :1]
        res /= 100.
        return res


class ClassifierMapper(DataMapper):

    def __init__(self, rgb_to_input, rgb_to_colorful_target_image):
        self.rgb_to_input = rgb_to_input
        self.rgb_to_colorful_target_image = rgb_to_colorful_target_image

    def class_to_color(self, cl):
        raise NotImplementedError('You need to implement a valid mapping from color-class to color space')

    def color_to_class(self, color):
        raise NotImplementedError('You need to implement a valid mapping from an input color space to a color-class')

    def rgb_to_target_image(self, rgb_image):
        return self.rgb_to_colorful_target_image(rgb_image)

    def rgb_to_colorizer_input(self, rgb_image):
        return self.rgb_to_input(rgb_image)

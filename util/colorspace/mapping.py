from __future__ import print_function

import numpy as np
from skimage.color import yuv2rgb, rgb2yuv, lab2rgb, rgb2lab


class DataMapper(object):

    def map(self, batch, mappings):
        if type(mappings) not in {list, tuple}:
            mappings = [mappings]
        res = []
        for i, mapping in enumerate(mappings):
            res.append([])
            for sample in batch:
                res[i].append(mapping(sample))
            res[i] = np.array(res[i])

        if len(res) == 1:   return res[0]
        else:               return res

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
        yuv = rgb2yuv(rgb_image.copy() / 255.)
        yuv[:, :, 1:] *= 2.
        return yuv

    def rgb_to_colorizer_target(self, rgb_image):
        return self.rgb_to_target_image(rgb_image)[:, :, 1:]

    def rgb_to_colorizer_input(self, rgb_image):
        yuv = rgb2yuv(rgb_image.copy() / 255.)
        res = yuv[:, :, :1]
        return res


class LabMapper(DataMapper):
    def network_prediction_to_rgb(self, prediction, inputs):
        lab = np.concatenate((inputs, prediction), axis=2)
        lab[:, :, :1] *= 100.
        lab[:, :, 1:] *= 128.
        return lab2rgb(lab)

    def rgb_to_target_image(self, rgb_image):
        lab = rgb2lab(rgb_image.copy() / 255.)
        lab[:, :, :1] /= 100.
        lab[:, :, 1:] /= 128.
        return lab

    def rgb_to_colorizer_target(self, rgb_image):
        return self.rgb_to_target_image(rgb_image)[:, :, 1:]

    def rgb_to_colorizer_input(self, rgb_image):
        lab = rgb2lab(rgb_image.copy() / 255.)
        res = lab[:, :, :1]
        res /= 100.
        return res


class LabClassMapper(LabMapper):
    def __init__(self, color_to_class, class_to_color, factor=9.):
        self.factor = factor
        self.color_to_class = color_to_class
        self.class_to_color = class_to_color
        self.nb_classes = len(class_to_color)

    def target_to_rgb(self, l, ab):
        ab = ab.astype(np.float32)
        ab *= self.factor
        lab = np.concatenate((l, ab), axis=2)
        lab[:, :, :1] *= 100.
        return lab2rgb(lab)

    def network_prediction_to_rgb(self, prediction, inputs):
        prediction = np.argmax(prediction, axis=2)
        ab = np.array([[self.class_to_color[clas] for clas in row] for row in prediction])
        return self.target_to_rgb(inputs, ab)

    def rgb_to_target_pairs(self, rgb_image):
        lab = rgb2lab(rgb_image.copy() / 255.)
        ab = lab[:, :, 1:]
        res = np.round(ab / self.factor) * self.factor
        return res.astype(np.int32)

    def rgb_to_classes(self, rgb_image):
        ab = self.rgb_to_target_pairs(rgb_image)
        return np.array([[self.color_to_class[tuple(color)] for color in row] for row in ab])

    def rgb_to_colorizer_target(self, rgb_image):
        ab = self.rgb_to_target_pairs(rgb_image)
        target_classes = np.array([[self.color_to_class[tuple(color)] for color in row] for row in ab])
        res = np.zeros(shape=target_classes.shape + (self.nb_classes,))
        for r in range(res.shape[0]):
            for c in range(res.shape[1]):
                target_color = ab[r][c]
                for i in range(-1, 1):
                    for j in range(-1, 1):
                        neighbour_color = (target_color[0] + int(i * self.factor),
                                           target_color[1] + int(j * self.factor))
                        if neighbour_color not in self.color_to_class:
                            continue
                        neighbour_class = self.color_to_class[neighbour_color]
                        if abs(i * j) == 1:     res[r][c][neighbour_class] = 0.1
                        elif i == 0 and j == 0: res[r][c][neighbour_class] = 0.3
                        else:                   res[r][c][neighbour_class] = 0.08
        return res


def get_mapper(color_space, classifier, color_to_class=None, class_to_color=None, factor=9.):
    color_space = color_space.lower()
    if classifier:
        if color_space == 'lab':        return LabClassMapper(color_to_class=color_to_class,
                                                              class_to_color=class_to_color,
                                                              factor=factor)
    else:
        if color_space == 'yuv':        return YUVMapper()
        elif color_space == 'lab':      return LabMapper()
    raise NotImplementedError('No implementation found for the specified color space')

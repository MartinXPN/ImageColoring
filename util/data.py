import math
import itertools

import numpy as np
from skimage.color import yuv2rgb, rgb2yuv, lab2rgb, rgb2lab
from tqdm import tqdm


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


class LabClassifierMapper(DataMapper):
    def __init__(self, color_to_class, class_to_color, factor=9.):
        self.factor = factor
        self.color_to_class = color_to_class
        self.class_to_color = class_to_color

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

    def rgb_to_target_image(self, rgb_image):
        lab = rgb2lab(rgb_image.copy() / 255.)
        lab[:, :, :1] /= 100.
        lab[:, :, 1:] /= 128.
        return lab

    def rgb_to_target_pairs(self, rgb_image):
        lab = rgb2lab(rgb_image.copy() / 255.)
        ab = lab[:, :, 1:]
        ab /= self.factor
        res = np.round(ab)
        return res.astype(np.int32)

    def rgb_to_colorizer_target(self, rgb_image):
        ab = self.rgb_to_target_pairs(rgb_image)
        res = np.array([[self.color_to_class[tuple(color)] for color in row] for row in ab])
        return np.expand_dims(res, axis=3)

    def rgb_to_colorizer_input(self, rgb_image):
        lab = rgb2lab(rgb_image.copy() / 255.)
        res = lab[:, :, :1]
        res /= 100.
        return res


class ColorMappingInitializer(object):
    def __init__(self, scale_factor=9.):
        self.color_to_class = {}
        self.class_to_color = []
        self.scale_factor = scale_factor

    def initialize(self):
        print('Initializing color space mappings...')
        self.class_to_color = []
        all_colors = self.generate_all_colors()
        for color in tqdm(all_colors):
            scaled_color = self.scale_color(color)
            if scaled_color not in self.color_to_class:
                self.color_to_class[scaled_color] = len(self.class_to_color)
                self.class_to_color.append(scaled_color)

        self.class_to_color = np.array(self.class_to_color)
        print(self.class_to_color.shape)

    def scale_color(self, color):
        return tuple([int(round(c / self.scale_factor)) for c in color])

    @staticmethod
    def generate_rgb_image_with_all_possible_values():
        all_colors = np.array(list(itertools.product(range(256), repeat=3)))
        image_edge = int(math.sqrt(all_colors.shape[0]))
        all_colors = all_colors.reshape((image_edge, image_edge, 3))
        return all_colors / 255.

    def generate_all_colors(self):
        all_lab = rgb2lab(self.generate_rgb_image_with_all_possible_values())
        ab_pairs = all_lab[:, :, 1:].reshape((all_lab.shape[0] * all_lab.shape[1], 2))
        print('All Colors shape:', all_lab.shape)
        print('Color pairs shape:', ab_pairs.shape)
        return ab_pairs

    def nb_classes(self):
        return len(self.class_to_color)


class ColorFrequencyCalculator(object):
    def __init__(self, color_to_class, class_to_color, data_mapper, image_generator, image_size):
        self.color_to_class = color_to_class
        self.class_to_color = class_to_color
        self.data_mapper = data_mapper
        self.image_generator = image_generator
        self.class_count = []
        self.image_size = image_size

        # self.pixel_class_count = {}
        # for i in range(image_size):
        #     self.pixel_class_count[i] = {}
        #     for j in range(image_size):
        #         self.pixel_class_count[i][j] = {}

    def populate_classes(self, rgb_image):
        target = self.data_mapper.rgb_to_target_pairs(rgb_image)
        for r in range(target.shape[0]):
            for c in range(target.shape[1]):
                color = tuple(target[r][c])
                if color not in self.color_to_class:
                    self.color_to_class[color] = len(self.class_to_color)
                    self.class_to_color.append(color)
                    self.class_count.append(0)
                color_class = self.color_to_class[color]
                self.class_count[color_class] += 1
                # if color_class not in self.pixel_class_count[r][c]:    self.pixel_class_count[r][c][color_class] = 1
                # else:                                                  self.pixel_class_count[r][c][color_class] += 1

    def populate(self, num_batches):
        print('Getting frequencies for every color-class from images...')
        for _ in tqdm(range(num_batches)):
            for rgb_image in next(self.image_generator):
                self.populate_classes(rgb_image)

    def get_class_weights(self):
        # shape = (self.image_size, self.image_size, len(self.class_to_color))
        # class_weights = np.zeros(shape=shape)
        # for row in range(shape[0]):
        #     for col in range(shape[1]):
        #         for clas in range(len(self.class_to_color)):
        #             if clas in self.pixel_class_count[row][col]:
        #                 class_weights[row][col][clas] = self.pixel_class_count[row][col][clas]
        counts = np.array(self.class_count)
        return np.broadcast_to(counts, shape=(self.image_size, self.image_size, len(counts)))


def get_mapper(color_space, classifier, color_to_class=None, class_to_color=None, factor=9.):
    color_space = color_space.lower()
    if classifier:
        if color_space == 'lab':        return LabClassifierMapper(color_to_class=color_to_class,
                                                                   class_to_color=class_to_color,
                                                                   factor=factor)
    else:
        if color_space == 'yuv':        return YUVMapper()
        elif color_space == 'lab':      return LabMapper()
    raise NotImplementedError('No implementation found for the specified color space')

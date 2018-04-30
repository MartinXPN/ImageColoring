import math
import itertools
import numpy as np

from tqdm import tqdm
from scipy.interpolate import interp1d
from skimage.color import rgb2lab


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
    def __init__(self, color_to_class, class_to_color, rgb_image_to_classes, image_generator, image_size):
        self.color_to_class = color_to_class
        self.class_to_color = class_to_color
        self.rgb_image_to_classes = rgb_image_to_classes
        self.image_generator = image_generator
        self.image_size = image_size

        shape = (self.image_size, self.image_size, len(self.class_to_color))
        self.pixel_class_count = np.zeros(shape=shape)
        self.class_count = np.zeros(shape=(len(self.class_to_color),))

    def populate_classes(self, rgb_image):
        image_classes = self.rgb_image_to_classes(rgb_image)
        for r in range(image_classes.shape[0]):
            for c in range(image_classes.shape[1]):
                color_class = image_classes[r][c]
                self.class_count[color_class] += 1
                self.pixel_class_count[r][c][color_class] += 1

    def populate(self, num_batches):
        print('Getting frequencies for every color-class from images...')
        for _ in tqdm(range(num_batches)):
            for rgb_image in next(self.image_generator):
                self.populate_classes(rgb_image)

    def get_class_weights(self):
        print('Calculating class weights...')
        for r in range(self.pixel_class_count.shape[0]):
            for c in range(self.pixel_class_count.shape[1]):
                m = interp1d(x=[-self.pixel_class_count[r][c].max(), -self.pixel_class_count[r][c].min()],
                             y=[0.5, 1.5])
                self.pixel_class_count[r][c] = m(-self.pixel_class_count[r][c])
        return self.pixel_class_count

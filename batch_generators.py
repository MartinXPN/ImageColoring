import random
import threading

import numpy as np
from keras.preprocessing.image import load_img

image_mean = (103.939, 116.779, 123.68)
greyscale_image_mean = np.mean(image_mean)


def unzip(iterable):
    return zip(*iterable)


class BaseBatchGenerator(object):
    
    def __init__(self, image_paths, batch_size, image_height, image_width, shuffle):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.lock = threading.Lock()
        self.index = 0
        if shuffle:
            random.shuffle(self.image_paths)

    def next(self):
        with self.lock:
            batch_inputs = []
            batch_labels = []

            for i in range(self.batch_size):
                path = self.image_paths[self.index]
                x, y = self.generate_one(path)
                
                batch_inputs.append(x)
                batch_labels.append(y)

                self.index += 1
                if self.index >= len(self.image_paths):
                    self.index = 0
            
            return self.get_reshaped_batch(batch_inputs, batch_labels)

    def load_gray_image(self, path):
        """
        :param path: path to image that needs to be loaded
        :return: 3-channeled greyscale image with size [self.image_height, self.image_width] and reduced mean
        """
        image_gray = load_img(path, grayscale=True, target_size=(self.image_height, self.image_width))
        res = np.zeros((self.image_height, self.image_width, 3))
        res[:, :, 0] = res[:, :, 1] = res[:, :, 2] = image_gray - greyscale_image_mean

        return res

    def generate_one(self, path):
        raise NotImplementedError("Please Implement this method")

    def get_reshaped_batch(self, batch_inputs, batch_labels):
        raise NotImplementedError("Please Implement this method")


class DiscriminatorRealGenerator(BaseBatchGenerator):
    """
    Generates real images to make the discriminator better in predicting real images
    """
    
    def __init__(self, image_paths, batch_size, image_height, image_width, shuffle=True):
        super(DiscriminatorRealGenerator, self).__init__(image_paths, batch_size, image_height, image_width, shuffle)
        
    def generate_one(self, path):
        image = load_img(path, target_size=(self.image_height, self.image_width, 3))
        image = np.array(image, dtype=np.float32)
        return (image, self.load_gray_image(path)), True

    def get_reshaped_batch(self, batch_inputs, batch_labels):
        batch_input_images, batch_gray_images = unzip(batch_inputs)
        print(np.array(batch_labels), 'Real images!')
        return [np.array(batch_input_images), np.array(batch_gray_images)], np.array(batch_labels)


class ColorizerBatchGenerator(BaseBatchGenerator):
    """
    Generates input for colorizer to improve it's performance in fooling the discriminator
    """
    def __init__(self, image_paths, batch_size, image_height, image_width, shuffle=True):
        super(ColorizerBatchGenerator, self).__init__(image_paths, batch_size, image_height, image_width, shuffle)

    def generate_one(self, path):
        target_image = self.get_generator_image_label(path, output_shape=(self.image_height, self.image_width))
        return self.load_gray_image(path), (self.get_label(), target_image)

    @staticmethod
    def get_generator_image_label(path, output_shape=(224, 224)):
        return np.array(load_img(path, target_size=output_shape), dtype=np.float32)

    def get_label(self):
        return True

    def get_reshaped_batch(self, batch_inputs, batch_labels):
        batch_labels, batch_image_labels = unzip(batch_labels)
        print(np.array(batch_labels))
        return np.array(batch_inputs), [np.array(batch_labels), np.array(batch_image_labels)]


class DiscriminatorFakeGenerator(ColorizerBatchGenerator):
    """
    Generates input for colorizer to improve discriminator's performance in detecting fake images
    """
    def __init__(self, image_paths, batch_size, image_height, image_width, shuffle=True):
        super(DiscriminatorFakeGenerator, self).__init__(image_paths, batch_size, image_height, image_width, shuffle)

    def get_label(self):
        return False

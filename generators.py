import os
from threading import Thread
from copy import copy

import numpy as np
from keras import backend as K
from keras.preprocessing.image import load_img, DirectoryIterator, img_to_array, array_to_img

from util.data import YUVMapper


class ImageDataGenerator(DirectoryIterator):
    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode=None,
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False):
        super(ImageDataGenerator, self).__init__(directory=directory, image_data_generator=image_data_generator,
                                                 target_size=target_size, color_mode=color_mode,
                                                 classes=classes, class_mode=class_mode,
                                                 batch_size=batch_size, shuffle=shuffle, seed=seed,
                                                 data_format=data_format,
                                                 save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format,
                                                 follow_links=follow_links)
        self.data_mapper = YUVMapper()
        self.generator_thread = Thread(target=self.generate_batch)
        self.generator_thread.start()
        self.batch = None

    def next(self):
        self.generator_thread.join()
        res = copy(self.batch)
        self.generator_thread = Thread(target=self.generate_batch)
        self.generator_thread.start()
        return res

    def generate_batch(self):
        with self.lock:
            index_array = next(self.index_generator)

        # The transformation of images is not under thread lock
        # so it can be done in parallel
        current_batch_size = len(index_array)
        grayscale = self.color_mode == 'grayscale'

        # build batch of image data
        batch_x = []
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname), grayscale=grayscale, target_size=self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.preprocessing_function(x)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x.append(x)
        batch_x = np.array(batch_x)

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, index in zip(range(current_batch_size), index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=index,
                                                                  hash=np.random.randint(10000),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        # build batch of labels
        if self.class_mode == 'sparse':         batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':       batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((current_batch_size, self.num_classes), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        elif self.class_mode == 'input':
            batch_y = np.zeros((current_batch_size,) + self.target_size + (3,), dtype=K.floatx())
            res = np.zeros((current_batch_size,) + self.target_size + (2,), dtype=K.floatx())
            for i, j in enumerate(index_array):
                img = load_img(os.path.join(self.directory, self.filenames[j]), target_size=self.target_size)
                batch_y[i] = img_to_array(img, data_format=self.data_format)
                res[i] = self.data_mapper.rgb_to_colorizer_target(batch_y[i])
            batch_y = res
        elif self.class_mode:
            batch_y = self.classes[index_array]
        else:
            batch_y = None

        res = tuple([item for item in [batch_x, batch_y] if item is not None])
        if len(res) == 1:
            res = res[0]
        self.batch = res
        return res

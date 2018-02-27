import os

import numpy as np
from keras import backend as K
from keras.preprocessing.image import load_img, DirectoryIterator, img_to_array, array_to_img


class ImageDataGenerator(DirectoryIterator):
    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode=None, label=None,
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
        self.label = label

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'

        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(10000),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        # build batch of labels
        if self.class_mode == 'input':          batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':       batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':       batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        elif self.class_mode == 'rgb':
            batch_y = np.zeros((current_batch_size,) + self.target_size + (3,), dtype=K.floatx())
            for i, j in enumerate(index_array):
                img = load_img(os.path.join(self.directory, self.filenames[j]), target_size=self.target_size)
                batch_y[i] = img_to_array(img, data_format=self.data_format)
        elif self.class_mode:
            batch_y = self.classes[index_array]
        else:
            if self.label is not None:
                return batch_x, np.array([self.label] * len(batch_x))
            return batch_x

        if self.label is not None:
            return batch_x, batch_y, np.array([self.label] * len(batch_x))
        return batch_x, batch_y

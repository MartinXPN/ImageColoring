from __future__ import print_function

import os

import fire
import keras
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imsave

from models.colorizer import get_colorizer
from util.data import get_mapper, ColorMappingInitializer


class Gym(object):
    def __init__(self,
                 colorizer, data_generator, data_mapper,
                 logger, models_save_dir, colored_images_save_dir):

        self.colorizer = colorizer
        self.data_generator = data_generator
        self.data_mapper = data_mapper

        self.model_save_dir = models_save_dir
        self.colored_images_save_dir = colored_images_save_dir
        if not os.path.exists(self.model_save_dir):             os.mkdir(self.model_save_dir)
        if not os.path.exists(self.colored_images_save_dir):    os.mkdir(self.colored_images_save_dir)

        self.logger = logger
        self.logger.set_model(self.colorizer)

    def train(self, epochs=100000, steps_per_epoch=500):
        batches = 0
        for epoch in range(epochs):
            self.evaluate(epoch=epoch)
            for step in range(steps_per_epoch):
                batches += 1
                rgb_images = next(self.data_generator)
                input_images, target_images = self.data_mapper.map(rgb_images, [self.data_mapper.rgb_to_colorizer_input,
                                                                                self.data_mapper.rgb_to_colorizer_target])
                loss = self.colorizer.train_on_batch(x=input_images, y=target_images)
                print('epoch: {}, step: {}, Loss: {}'.format(epoch, step, loss))
                self.logger.on_epoch_end(epoch=batches, logs={'train loss': loss})

    def evaluate(self, epoch):
        print('Evaluating epoch {} ...'.format(epoch), end='\t')
        rgb_images = next(self.data_generator)
        input_images = self.data_mapper.map(rgb_images, self.data_mapper.rgb_to_colorizer_input)
        colored_images = self.colorizer.predict(input_images)

        for i, image in enumerate(colored_images):
            rgb_prediction = self.data_mapper.network_prediction_to_rgb(prediction=colored_images[i], inputs=input_images[i])
            imsave(name=os.path.join(self.colored_images_save_dir, 'epoch-{}-{}-colored.jpg'.format(epoch, i)), arr=rgb_prediction)
            imsave(name=os.path.join(self.colored_images_save_dir, 'epoch-{}-{}-target.jpg'.format(epoch, i)), arr=rgb_images[i])
        self.colorizer.save(filepath=os.path.join(self.model_save_dir, 'epoch={}.hdf5'.format(epoch)))
        print('Done!')


def main(batch_size=32, image_size=224, epochs=100000, steps_per_epoch=100, color_space='yuv',
         train_data_dir='/mnt/bolbol/raw-data/train', valid_data_dir='/mnt/bolbol/raw-data/validation',
         log_dir='logs', models_save_dir='coloring_models', colored_images_save_dir='colored_images',
         classifier=False, populate_batches=1000,
         vgg=False, feature_extractor_model_path=None, train_feature_extractor=False):
    """ Train only colorizer on target images """

    ''' Prepare data generators '''
    train_generator = ImageDataGenerator().flow_from_directory(directory=train_data_dir,
                                                               target_size=(image_size, image_size),
                                                               batch_size=batch_size,
                                                               color_mode='rgb',
                                                               class_mode=None)
    nb_color_classes = 0
    if classifier:
        data_mapper = get_mapper(color_space, classifier)
        mapping = ColorMappingInitializer()
        mapping.initialize()
        nb_color_classes = mapping.nb_classes()
        data_mapper = get_mapper(color_space=color_space, classifier=classifier,
                                 color_to_class=mapping.color_to_class, class_to_color=mapping.class_to_color,
                                 factor=mapping.scale_factor)
    else:
        data_mapper = get_mapper(color_space=color_space, classifier=classifier)

    ''' Prepare Models '''
    colorizer = get_colorizer(image_size=image_size, vgg=vgg, feature_extractor_model_path=feature_extractor_model_path,
                              train_feature_extractor=train_feature_extractor,
                              classifier=classifier,
                              classes_per_pixel=nb_color_classes)
    colorizer.compile(optimizer=Adam(lr=3e-4), loss='sparse_categorical_crossentropy' if classifier else 'mse')

    ''' View summary of the models '''
    print('\n\n\n\nColorizer:'),    colorizer.summary()

    logger = keras.callbacks.TensorBoard(log_dir=log_dir) if K.backend() == 'tensorflow' else Callback()
    gym = Gym(colorizer=colorizer,
              data_generator=train_generator,
              data_mapper=data_mapper,
              logger=logger,
              models_save_dir=models_save_dir,
              colored_images_save_dir=colored_images_save_dir)
    gym.train(epochs=epochs, steps_per_epoch=steps_per_epoch)


if __name__ == '__main__':
    fire.Fire(main)

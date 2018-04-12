from __future__ import print_function

import os

import fire
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imsave

import generators
from models.colorizer import Colorizer, VGGColorizer
from util.data import get_mapper


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
                input_images, target_images = self.data_generator.next()
                loss = self.colorizer.train_on_batch(x=input_images, y=target_images)
                print('epoch: {}, step: {}, Loss: {}'.format(epoch, step, loss))
                self.logger.on_epoch_end(epoch=batches, logs={'train loss': loss})

    def evaluate(self, epoch):
        print('Evaluating epoch {} ...'.format(epoch), end='\t')
        input_images, target_images = self.data_generator.next()
        colored_images = self.colorizer.predict(input_images)
        for i, image in enumerate(colored_images):
            rgb_prediction = self.data_mapper.network_prediction_to_rgb(colored_images[i], input_images[i])
            imsave(name=os.path.join(self.colored_images_save_dir, str(epoch) + '-' + str(i) + '.jpg'),
                   arr=rgb_prediction)
        self.colorizer.save(os.path.join(self.model_save_dir, 'epoch={}.hdf5'.format(epoch)))
        print('Done!')


def main(batch_size=32, image_size=224, epochs=100000, steps_per_epoch=100, color_space='yuv',
         train_data_dir='/mnt/bolbol/raw-data/train', valid_data_dir='/mnt/bolbol/raw-data/validation',
         log_dir='logs', models_save_dir='coloring_models', colored_images_save_dir='colored_images',
         vgg=False, feature_extractor_model_path=None, train_feature_extractor=False):
    """ Train only colorizer on target images """
    data_mapper = get_mapper(color_space)

    ''' Prepare Models '''
    if not vgg:     colorizer = Colorizer(input_shape=(image_size, image_size, 1))
    else:           colorizer = VGGColorizer(input_shape=(image_size, image_size, 1),
                                             feature_extractor_model_path=feature_extractor_model_path,
                                             train_feature_extractor=train_feature_extractor)
    colorizer.compile(optimizer=Adam(lr=3e-4), loss='mse')

    ''' View summary of the models '''
    print('\n\n\n\nColorizer:'),    colorizer.summary()

    ''' Prepare data generators '''
    generator = ImageDataGenerator(preprocessing_function=data_mapper.rgb_to_colorizer_input)
    train_generator = generators.ImageDataGenerator(directory=train_data_dir,
                                                    image_data_generator=generator,
                                                    target_size=(image_size, image_size),
                                                    batch_size=batch_size,
                                                    color_mode='rgb',
                                                    class_mode='input')

    logger = TensorBoard(log_dir=log_dir)
    gym = Gym(colorizer=colorizer,
              data_generator=train_generator,
              data_mapper=data_mapper,
              logger=logger,
              models_save_dir=models_save_dir,
              colored_images_save_dir=colored_images_save_dir)
    gym.train(epochs=epochs, steps_per_epoch=steps_per_epoch)


if __name__ == '__main__':
    fire.Fire(main)

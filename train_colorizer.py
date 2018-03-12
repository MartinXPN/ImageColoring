from __future__ import print_function

import argparse
import os

from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

import generators
from models.colorizer import Colorizer
from util.data import rgb_to_colorizer_input


class Gym(object):
    def __init__(self,
                 colorizer, data_generator,
                 logger, models_save_dir, colored_images_save_dir):

        self.colorizer = colorizer
        self.data_generator = data_generator

        self.model_save_dir = models_save_dir
        self.colored_images_save_dir = colored_images_save_dir
        if not os.path.exists(self.model_save_dir):             os.mkdir(self.model_save_dir)
        if not os.path.exists(self.colored_images_save_dir):    os.mkdir(self.colored_images_save_dir)

        self.logger = logger
        self.logger.set_model(self.colorizer)

    def train(self, epochs=100000):

        for epoch in tqdm(range(epochs)):
            fool_inputs, target_images = self.data_generator.next()
            loss = self.colorizer.train_on_batch(x=fool_inputs, y=target_images)
            self.logger.on_epoch_end(epoch=epoch, logs={'train loss': loss})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',         default=32,     help='Batch size',                          type=int)
    parser.add_argument('--image_size',         default=224,    help='Batch size',                          type=int)
    parser.add_argument('--train_data_dir',     default='/mnt/bolbol/raw-data/train',                       type=str)
    parser.add_argument('--valid_data_dir',     default='/mnt/bolbol/raw-data/validation',                  type=str)
    parser.add_argument('--logdir',             default='./logs',       help='Where to log the progres',    type=str)
    parser.add_argument('--models_save_dir',    default='coloring_models',  help='Where to save models',    type=str)
    parser.add_argument('--eval_images_dir',    default='colored_images',   help='Where to save images',    type=str)
    parser.add_argument('--feature_extractor_model_path',
                        default='finetune-70-2.15-no-top.hdf5',
                        help='Path to VGG/Feature extractor model or weights')
    args = parser.parse_args()

    ''' Prepare Models '''
    colorizer = Colorizer(feature_extractor_model_path=args.feature_extractor_model_path,
                          input_shape=(args.image_size, args.image_size, 1))
    colorizer.compile(optimizer='adam', loss='mae')

    ''' View summary of the models '''
    print('\n\n\n\nColorizer:'),    colorizer.summary()

    ''' Prepare data generators '''
    generator = ImageDataGenerator(preprocessing_function=rgb_to_colorizer_input)
    train_generator = generators.ImageDataGenerator(directory=args.train_data_dir,
                                                    image_data_generator=generator,
                                                    target_size=(args.image_size, args.image_size),
                                                    batch_size=args.batch_size,
                                                    color_mode='rgb',
                                                    class_mode='input')

    logger = TensorBoard(log_dir=args.logdir)
    gym = Gym(colorizer=colorizer,
              data_generator=train_generator,
              logger=logger,
              models_save_dir=args.models_save_dir,
              colored_images_save_dir=args.eval_images_dir)
    gym.train()


if __name__ == '__main__':
    main()

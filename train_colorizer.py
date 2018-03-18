from __future__ import print_function

import argparse
import os

from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imsave
from tqdm import tqdm

import generators
from models.colorizer import Colorizer
from util.data import rgb_to_colorizer_input, network_prediction_to_rgb


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
        self.input_images, self.target_images = self.data_generator.next()  # TODO delete

    def train(self, epochs=100000, steps_per_epoch=500):
        batches = 0
        for epoch in range(epochs):
            self.evaluate(epoch=epoch)
            for step in range(steps_per_epoch):
                batches += 1
                # input_images, target_images = self.data_generator.next()
                loss = self.colorizer.train_on_batch(x=self.input_images, y=self.target_images)
                print('epoch: {}, step: {}, Loss: {}'.format(epoch, step, loss))
                self.logger.on_epoch_end(epoch=batches, logs={'train loss': loss})

    def evaluate(self, epoch):
        print('Evaluating epoch {} ...'.format(epoch), end='\t')
        # input_images, target_images = self.data_generator.next()
        colored_images = self.colorizer.predict(self.input_images)
        for i, image in enumerate(colored_images):
            rgb_prediction = network_prediction_to_rgb(colored_images[i], self.input_images[i])
            imsave(name=os.path.join(self.colored_images_save_dir, str(epoch) + '-' + str(i) + '.jpg'),
                   arr=rgb_prediction)
        self.colorizer.save(os.path.join(self.model_save_dir, 'epoch={}.hdf5'.format(epoch)))
        print('Done!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',         default=8,      help='Batch size',                          type=int)
    parser.add_argument('--image_size',         default=224,    help='Image size',                          type=int)
    parser.add_argument('--epochs',             default=100000, help='Number of epochs',                    type=int)
    parser.add_argument('--steps_per_epoch',    default=500,    help='Number of batches per one epoch',     type=int)
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
    colorizer.compile(optimizer=Adam(lr=3e-4), loss='mse')

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
    gym.train(epochs=args.epochs, steps_per_epoch=args.steps_per_epoch)


if __name__ == '__main__':
    main()

from __future__ import print_function

import argparse
import os

import numpy as np
from scipy.misc import imsave
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator

import generators
from models.colorizer import Colorizer
from models.critic import Critic
from models.gan import CombinedGan
from util.data import rgb_to_colorizer_input, rgb_to_target_image, network_prediction_to_rgb


def wasserstein_loss(target, output):
    return K.mean(target * output)


class Gym(object):
    def __init__(self,
                 generator, critic, combined,
                 generator_data_generator, real_data_generator, combined_data_generator,
                 logger, models_save_dir, colored_images_save_dir):

        self.generator = generator
        self.critic = critic
        self.combined = combined

        ''' Data '''
        self.generator_data_generator = generator_data_generator
        self.real_data_generator = real_data_generator
        self.combined_data_generator = combined_data_generator

        self.model_save_dir = models_save_dir
        self.colored_images_save_dir = colored_images_save_dir
        if not os.path.exists(self.model_save_dir):             os.mkdir(self.model_save_dir)
        if not os.path.exists(self.colored_images_save_dir):    os.mkdir(self.colored_images_save_dir)

        self.logger = logger
        self.logger.set_model(self.combined)

    def train(self, loss_threshold=-0.1, eval_interval=100, epochs=100000):

        def train_critic_real():
            # Train critic on real data
            train_critic_real.steps += 1
            real_images = self.real_data_generator.next()
            real_labels = -np.ones(shape=len(real_images))

            loss = self.critic.train_on_batch(x=real_images, y=real_labels)
            self.logger.on_epoch_end(epoch=train_critic_real.steps,
                                     logs={'Critic loss on real data': loss})
            print('Loss on real data:', loss)
            return loss

        def train_critic_fake():
            # Train critic on fake data
            train_critic_fake.steps += 1
            gray = self.generator_data_generator.next()
            colors = self.generator.predict(gray)
            fake_images = np.concatenate((gray, colors), axis=3)
            fake_labels = np.ones(shape=len(colors))

            loss = self.critic.train_on_batch(x=fake_images, y=fake_labels)
            self.logger.on_epoch_end(epoch=train_critic_fake.steps,
                                     logs={'Critic loss on fake data': loss})
            print('Loss on fake data:', loss)
            return loss

        def train_generator_fool_critic():
            # Train generator to fool the critic
            train_generator_fool_critic.steps += 1
            fool_inputs, target_images = self.combined_data_generator.next()
            fool_labels = -np.ones(shape=len(fool_inputs))

            # [_, loss, l1_loss] = self.combined.train_on_batch(x=fool_inputs, y=[fool_labels, target_images])
            # self.logger.on_epoch_end(epoch=train_generator_fool_critic.steps,
            #                          logs={'Target image difference loss': l1_loss, 'Fool critic loss': loss})
            loss = self.combined.train_on_batch(x=fool_inputs, y=fool_labels)
            self.logger.on_epoch_end(epoch=train_generator_fool_critic.steps,
                                     logs={'Fool critic loss': loss})
            print('Fool loss: ', loss)
            return loss

        ''' Initialize counters '''
        train_critic_real.steps = 0
        train_critic_fake.steps = 0
        train_generator_fool_critic.steps = 0

        ''' Start training '''
        for epoch in range(epochs):
            while train_generator_fool_critic() > loss_threshold:   pass
            while train_critic_real() > loss_threshold:             pass
            while train_critic_fake() > loss_threshold:             pass
            if epoch % eval_interval == 0:
                self.evaluate(epoch=epoch)

    def evaluate(self, epoch):
        print('Evaluating epoch {} ...'.format(epoch), end='\t')
        greyscale_images = self.generator_data_generator.next()
        colored_images = self.generator.predict(greyscale_images)
        for i, image in enumerate(colored_images):
            rgb_prediction = network_prediction_to_rgb(colored_images[i], greyscale_images[i])
            imsave(name=os.path.join(self.colored_images_save_dir, str(epoch) + '-' + str(i) + '.jpg'),
                   arr=rgb_prediction)
        self.generator.save(os.path.join(self.model_save_dir, 'epoch={}.hdf5'.format(epoch)))
        print('Done!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',         default=64,     help='Batch size',                          type=int)
    parser.add_argument('--image_size',         default=224,    help='Batch size',                          type=int)
    parser.add_argument('--epoch_images',       default=5000,   help='Number of images seen in one epoch',  type=int)
    parser.add_argument('--loss_threshold',     default=-0.1,   help='Switch to next training step',        type=float)
    parser.add_argument('--train_data_dir',     default='/mnt/bolbol/raw-data/train',                       type=str)
    parser.add_argument('--valid_data_dir',     default='/mnt/bolbol/raw-data/validation',                  type=str)
    parser.add_argument('--logdir',             default='./logs',   help='Where to log the progres',        type=str)
    parser.add_argument('--models_save_dir',    default='coloring_models',  help='Where to save models',    type=str)
    parser.add_argument('--eval_images_dir',    default='colored_images',   help='Where to save images',    type=str)
    parser.add_argument('--feature_extractor_model_path',
                        default='finetune-70-2.15-no-top.hdf5',
                        help='Path to VGG/Feature extractor model or weights')
    args = parser.parse_args()

    ''' Prepare Models '''
    colorizer = Colorizer(feature_extractor_model_path=args.feature_extractor_model_path,
                          input_shape=(args.image_size, args.image_size, 1))
    critic = Critic(input_shape=(args.image_size, args.image_size, 3))
    critic.compile(optimizer=RMSprop(lr=0.00005), loss=wasserstein_loss)
    combined = CombinedGan(generator=colorizer, critic=critic,
                           input_shape=(args.image_size, args.image_size, 1), include_colorizer_output=False)
    combined.compile(optimizer=Adam(lr=3e-4), loss=[wasserstein_loss])

    ''' View summary of the models '''
    print('\n\n\n\nColorizer:'),    colorizer.summary()
    print('\n\n\n\nCritic:'),       critic.summary()
    print('\n\n\n\nCombined:'),     combined.summary()

    ''' Prepare data generators '''
    greyscale_generator = ImageDataGenerator(preprocessing_function=rgb_to_colorizer_input)
    real_data_generator = ImageDataGenerator(preprocessing_function=rgb_to_target_image)
    combined_generator  = ImageDataGenerator(preprocessing_function=rgb_to_colorizer_input)
    greyscale_generator = generators.ImageDataGenerator(directory=args.train_data_dir,
                                                        image_data_generator=greyscale_generator,
                                                        target_size=(args.image_size, args.image_size),
                                                        batch_size=args.batch_size,
                                                        color_mode='rgb')
    real_data_generator = generators.ImageDataGenerator(directory=args.train_data_dir,
                                                        image_data_generator=real_data_generator,
                                                        target_size=(args.image_size, args.image_size),
                                                        batch_size=args.batch_size,
                                                        color_mode='rgb')
    combined_generator = generators.ImageDataGenerator(directory=args.train_data_dir,
                                                       image_data_generator=combined_generator,
                                                       target_size=(args.image_size, args.image_size),
                                                       batch_size=args.batch_size,
                                                       color_mode='rgb',
                                                       class_mode='input')

    logger = TensorBoard(log_dir=args.logdir)
    gym = Gym(generator=colorizer, critic=critic, combined=combined,
              generator_data_generator=greyscale_generator,
              real_data_generator=real_data_generator,
              combined_data_generator=combined_generator,
              logger=logger,
              models_save_dir=args.models_save_dir,
              colored_images_save_dir=args.eval_images_dir)
    gym.train(loss_threshold=args.loss_threshold)


if __name__ == '__main__':
    main()

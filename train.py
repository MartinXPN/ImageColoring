import argparse
import os

import numpy as np
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator

import generators
from models.colorizer import Colorizer
from models.critic import Critic
from models.gan import CombinedGan


def wasserstein_loss(target, output):
    return K.mean(target * output)


class Gym(object):
    def __init__(self,
                 generator, critic, combined,
                 generator_data_generator, real_data_generator, combined_data_generator,
                 logger,
                 models_save_dir):

        self.generator = generator
        self.critic = critic
        self.combined = combined

        ''' Data '''
        self.generator_data_generator = generator_data_generator
        self.real_data_generator = real_data_generator
        self.combined_data_generator = combined_data_generator

        self.model_save_dir = models_save_dir
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

        self.logger = logger
        self.logger.set_model(self.combined)

    def train(self):

        def train_critic_real():
            # Train critic on real data
            train_critic_real.steps += 1
            real_images = self.real_data_generator.next()
            real_labels = np.array([-1] * len(real_images))

            loss = self.critic.train_on_batch(x=real_images, y=real_labels)
            self.logger.on_epoch_end(epoch=train_critic_real.steps,
                                     logs={'Critic loss on real data': loss})
            return loss

        def train_critic_fake():
            # Train critic on fake data
            train_critic_fake.steps += 1
            greyscale_images = self.generator_data_generator.next()
            fake_images = self.generator.predict(greyscale_images)
            fake_labels = np.array([1] * len(fake_images))

            loss = self.critic.train_on_batch(x=fake_images, y=fake_labels)
            self.logger.on_epoch_end(epoch=train_critic_fake.steps,
                                     logs={'Critic loss on fake data': loss})
            return loss

        def train_generator_fool_critic():
            # Train generator to fool the critic
            train_generator_fool_critic.steps += 1
            fool_inputs, target_images = self.combined_data_generator.next()
            fool_labels = np.array([-1] * len(fool_inputs))

            [loss, l1_loss, _] = self.combined.train_on_batch(x=fool_inputs, y=[fool_labels, target_images])
            self.logger.on_epoch_end(epoch=train_generator_fool_critic.steps,
                                     logs={'Target image L1 loss': l1_loss, 'Combined prediction loss': loss})
            return loss

        ''' Initialize counters '''
        train_critic_real.steps = 0
        train_critic_fake.steps = 0
        train_generator_fool_critic.steps = 0

        ''' Start training '''
        while True:
            while train_critic_real() > 0.1:            pass
            while train_critic_fake() > 0.1:            pass
            while train_generator_fool_critic() > 0.1:  pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',         default=5,      help='Batch size',                          type=int)
    parser.add_argument('--image_size',         default=224,    help='Batch size',                          type=int)
    parser.add_argument('--epoch_images',       default=5000,   help='Number of images seen in one epoch',  type=int)
    parser.add_argument('--train_data_dir',     default='/mnt/bolbol/raw-data/train',                       type=str)
    parser.add_argument('--valid_data_dir',     default='/mnt/bolbol/raw-data/validation',                  type=str)
    parser.add_argument('--logdir',             default='./logs',   help='Where to log the progres',        type=str)
    parser.add_argument('--models_save_dir',    default='coloring_models',  help='Where to save models',    type=str)
    parser.add_argument('--feature_extractor_model_path',
                        default='/Users/martin/Desktop/finetune-40-2.08-no-top.hdf5',
                        help='Path to VGG/Feature extractor model or weights')
    args = parser.parse_args()

    ''' Prepare Models '''
    colorizer = Colorizer(feature_extractor_model_path=args.feature_extractor_model_path,
                          input_shape=(args.image_size, args.image_size, 1))
    critic = Critic(input_shape=(args.image_size, args.image_size, 3))
    combined = CombinedGan(generator=colorizer, critic=critic, input_shape=(args.image_size, args.image_size, 1))
    critic.compile(optimizer='adam', loss=wasserstein_loss)
    combined.compile(optimizer='rmsprop', loss=[wasserstein_loss, 'mae'])
    combined.summary()

    ''' Prepare data generators '''
    generator = ImageDataGenerator(preprocessing_function=lambda x: (x - 128.) / 128.)
    greyscale_generator = generators.ImageDataGenerator(directory=args.train_data_dir,
                                                        image_data_generator=generator,
                                                        target_size=(args.image_size, args.image_size),
                                                        batch_size=args.batch_size,
                                                        color_mode='grayscale')
    real_data_generator = generators.ImageDataGenerator(directory=args.train_data_dir,
                                                        image_data_generator=generator,
                                                        target_size=(args.image_size, args.image_size),
                                                        batch_size=args.batch_size,
                                                        color_mode='rgb')
    combined_generator = generators.ImageDataGenerator(directory=args.train_data_dir,
                                                       image_data_generator=generator,
                                                       target_size=(args.image_size, args.image_size),
                                                       batch_size=args.batch_size,
                                                       color_mode='grayscale',
                                                       class_mode='input')

    logger = TensorBoard(log_dir=args.logdir)
    gym = Gym(generator=colorizer, critic=critic, combined=combined,
              generator_data_generator=greyscale_generator,
              real_data_generator=real_data_generator,
              combined_data_generator=combined_generator,
              logger=logger,
              models_save_dir=args.models_save_dir)
    gym.train()


if __name__ == '__main__':
    main()

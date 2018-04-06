from __future__ import print_function

import os

import fire
import keras
import numpy as np
from keras import backend as K
from keras.callbacks import Callback
from keras.models import load_model
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imsave

import generators
from models.colorizer import Colorizer, VGGColorizer
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

    def train(self, loss_threshold=-0.1, eval_interval=10, epochs=100000, include_target_image=False):

        def train_critic_real():
            """ Train critic on real data """
            train_critic_real.steps += 1
            real_images = self.real_data_generator.next()
            real_labels = -np.ones(shape=len(real_images))

            loss = self.critic.train_on_batch(x=real_images, y=real_labels)
            self.logger.on_epoch_end(epoch=train_critic_real.steps,
                                     logs={'Critic loss on real data': loss})
            print('Loss on real data:', loss)
            return loss

        def train_critic_fake():
            """ Train critic on fake data """
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
            """ Train generator to fool the critic """
            train_generator_fool_critic.steps += 1
            fool_inputs, target_images = self.combined_data_generator.next()
            fool_labels = -np.ones(shape=len(fool_inputs))

            # [sum, loss, l1_loss] or loss
            loss = self.combined.train_on_batch(x=fool_inputs,
                                                y=[fool_labels, target_images] if include_target_image else fool_labels)
            self.logger.on_epoch_end(epoch=train_generator_fool_critic.steps,
                                     logs={'Fool critic loss': loss[1], 'Target image difference loss': loss[2]}  if include_target_image else
                                          {'Fool critic loss': loss})
            print('Fool loss: ', loss)
            return loss

        ''' Initialize counters '''
        train_critic_real.steps = 0
        train_critic_fake.steps = 0
        train_generator_fool_critic.steps = 0

        ''' Start training '''
        for epoch in range(epochs):
            ''' Train critic '''
            real_loss = fake_loss = 10
            while real_loss > loss_threshold or fake_loss > loss_threshold:
                while fake_loss > real_loss + 0.2:      fake_loss = train_critic_fake()
                while real_loss > fake_loss + 0.2:      real_loss = train_critic_real()
                fake_loss = train_critic_fake()
                real_loss = train_critic_real()

            ''' Train colorizer '''
            while train_generator_fool_critic() > loss_threshold:
                pass

            ''' Evaluate '''
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


def main(batch_size=32, eval_interval=10, epochs=100000, image_size=224, loss_threshold=-0.1,
         train_data_dir='/mnt/bolbol/raw-data/train',
         log_dir='logs', models_save_dir='coloring_models', colored_images_save_dir='colored_images',
         vgg=False, feature_extractor_model_path=None, train_feature_extractor=False,
         colorizer_model_path=None,
         include_target_image=False):
    """ Train Wasserstein gan to colorize black and white images """

    ''' Prepare Models '''
    if colorizer_model_path:    colorizer = load_model(filepath=colorizer_model_path, custom_objects={'Colorizer': Colorizer}, compile=False)
    elif not vgg:               colorizer = Colorizer(input_shape=(image_size, image_size, 1))
    else:                       colorizer = VGGColorizer(input_shape=(image_size, image_size, 1),
                                                         feature_extractor_model_path=feature_extractor_model_path,
                                                         train_feature_extractor=train_feature_extractor)
    critic = Critic(input_shape=(image_size, image_size, 3))
    critic.compile(optimizer=RMSprop(lr=0.00005), loss=wasserstein_loss)
    combined = CombinedGan(generator=colorizer, critic=critic,
                           input_shape=(image_size, image_size, 1),
                           include_colorizer_output=include_target_image)
    combined.compile(optimizer=Adam(lr=3e-4),
                     loss=[wasserstein_loss, 'mse'] if include_target_image else [wasserstein_loss])

    ''' View summary of the models '''
    print('\n\n\n\nColorizer:'),    colorizer.summary()
    print('\n\n\n\nCritic:'),       critic.summary()
    print('\n\n\n\nCombined:'),     combined.summary()

    ''' Prepare data generators '''
    greyscale_generator = ImageDataGenerator(preprocessing_function=rgb_to_colorizer_input)
    real_data_generator = ImageDataGenerator(preprocessing_function=rgb_to_target_image)
    combined_generator  = ImageDataGenerator(preprocessing_function=rgb_to_colorizer_input)
    greyscale_generator = generators.ImageDataGenerator(directory=train_data_dir,
                                                        image_data_generator=greyscale_generator,
                                                        target_size=(image_size, image_size),
                                                        batch_size=batch_size,
                                                        color_mode='rgb')
    real_data_generator = generators.ImageDataGenerator(directory=train_data_dir,
                                                        image_data_generator=real_data_generator,
                                                        target_size=(image_size, image_size),
                                                        batch_size=batch_size,
                                                        color_mode='rgb')
    combined_generator = generators.ImageDataGenerator(directory=train_data_dir,
                                                       image_data_generator=combined_generator,
                                                       target_size=(image_size, image_size),
                                                       batch_size=batch_size,
                                                       color_mode='rgb',
                                                       class_mode='input')

    logger = keras.callbacks.TensorBoard(log_dir=log_dir) if K.backend() == 'tensorflow' else Callback()
    gym = Gym(generator=colorizer, critic=critic, combined=combined,
              generator_data_generator=greyscale_generator,
              real_data_generator=real_data_generator,
              combined_data_generator=combined_generator,
              logger=logger,
              models_save_dir=models_save_dir,
              colored_images_save_dir=colored_images_save_dir)
    gym.train(loss_threshold=loss_threshold,
              eval_interval=eval_interval, epochs=epochs,
              include_target_image=include_target_image)


if __name__ == '__main__':
    fire.Fire(main)

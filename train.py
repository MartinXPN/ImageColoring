from __future__ import print_function

import os
import fire
import keras
import numpy as np

from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imsave

from models.colorizer import get_colorizer
from models.critic import Critic
from models.gan import get_combined_gan
from util.colorspace.initialize import get_mapping_with_class_weights


def wasserstein_loss(target, output):
    return K.mean(target * output)


class Gym(object):
    def __init__(self,
                 generator, critic, combined,
                 image_generator, data_mapper, logger,
                 models_save_dir, colored_images_save_dir,
                 classifier=False):

        self.generator = generator
        self.critic = critic
        self.combined = combined

        ''' Data '''
        self.image_generator = image_generator
        self.data_mapper = data_mapper
        self.class_to_color = data_mapper.class_to_color
        self.classifier = classifier

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
            rgb_images = next(self.image_generator)
            real_images = self.data_mapper.map(rgb_images, self.data_mapper.rgb_to_target_image)
            real_labels = -np.ones(shape=len(real_images))

            loss = self.critic.train_on_batch(x=real_images, y=real_labels)
            self.logger.on_epoch_end(epoch=train_critic_real.steps,
                                     logs={'Critic loss on real data': loss})
            print('Loss on real data:', loss)
            return loss

        def train_critic_fake():
            """ Train critic on fake data """
            train_critic_fake.steps += 1
            rgb_images = next(self.image_generator)
            gray = self.data_mapper.map(rgb_images, self.data_mapper.rgb_to_colorizer_input)
            colors = self.generator.predict(gray)
            colors = np.dot(colors, self.class_to_color) if self.classifier else colors  # map classes to colors
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
            rgb_images = next(self.image_generator)
            fool_inputs, target_images = self.data_mapper.map(rgb_images, [self.data_mapper.rgb_to_colorizer_input,
                                                                           self.data_mapper.rgb_to_colorizer_target])
            fool_labels = -np.ones(shape=len(fool_inputs))

            # [sum, loss, l1_loss] or loss
            loss = self.combined.train_on_batch(x=fool_inputs,
                                                y=[fool_labels, target_images] if include_target_image else fool_labels)
            self.logger.on_epoch_end(epoch=train_generator_fool_critic.steps,
                                     logs={'Fool critic loss': loss[1], 'Target image difference loss': loss[2]}  if include_target_image else
                                          {'Fool critic loss': loss})
            print('Fool loss: ', loss)
            return loss[1] if include_target_image else loss

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
        rgb_images = next(self.image_generator)
        print(rgb_images.shape)
        input_images = self.data_mapper.map(rgb_images, self.data_mapper.rgb_to_colorizer_input)
        colored_images = self.generator.predict(input_images)

        for i, image in enumerate(colored_images):
            rgb_prediction = self.data_mapper.network_prediction_to_rgb(prediction=colored_images[i], inputs=input_images[i])
            imsave(name=os.path.join(self.colored_images_save_dir, 'epoch-{}-{}-colored.jpg'.format(epoch, i)), arr=rgb_prediction)
            imsave(name=os.path.join(self.colored_images_save_dir, 'epoch-{}-{}-target.jpg'.format(epoch, i)), arr=rgb_images[i])
        self.generator.save(filepath=os.path.join(self.model_save_dir, 'epoch={}.hdf5'.format(epoch)))
        print('Done!')


def main(batch_size=32, eval_interval=10, epochs=100000, image_size=224, loss_threshold=-0.1, color_space='yuv',
         train_data_dir='/mnt/bolbol/raw-data/train',
         log_dir='logs', models_save_dir='coloring_models', colored_images_save_dir='colored_images',
         vgg=False, feature_extractor_model_path=None, train_feature_extractor=False,
         classifier=False, populate_batches=1000, scale_factor=9.,
         colorizer_model_path=None,
         include_target_image=False):
    """ Train Wasserstein gan to colorize black and white images """
    ''' Prepare data generators '''
    image_generator = ImageDataGenerator().flow_from_directory(directory=train_data_dir,
                                                               target_size=(image_size, image_size),
                                                               batch_size=batch_size,
                                                               color_mode='rgb', class_mode=None)
    data_mapper, class_weights = get_mapping_with_class_weights(classifier=classifier, color_space=color_space,
                                                                image_generator=image_generator, image_size=image_size,
                                                                nb_batches=populate_batches, scale_factor=scale_factor)

    ''' Prepare Models '''
    colorizer = get_colorizer(colorizer_model_path=colorizer_model_path, image_size=image_size,
                              vgg=vgg, feature_extractor_model_path=feature_extractor_model_path,
                              train_feature_extractor=train_feature_extractor,
                              classifier=classifier, classes_per_pixel=data_mapper.nb_classes if classifier else 0)
    critic = Critic(input_shape=(image_size, image_size, 3))
    critic.compile(optimizer=Adam(lr=0.00001, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)
    combined = get_combined_gan(classifier=classifier, color_space=color_space, generator=colorizer, critic=critic,
                                input_shape=(image_size, image_size, 1),
                                include_colorizer_output=include_target_image,
                                class_to_color=data_mapper.class_to_color)
    combined.compile(optimizer=Adam(lr=0.00001, beta_1=0.5, beta_2=0.9),
                     loss=[wasserstein_loss, 'categorical_crossentropy' if classifier else 'mse'] if include_target_image
                     else [wasserstein_loss])

    ''' View summary of the models '''
    print('\n\n\n\nColorizer:'),    colorizer.summary()
    print('\n\n\n\nCritic:'),       critic.summary()
    print('\n\n\n\nCombined:'),     combined.summary()

    logger = keras.callbacks.TensorBoard(log_dir=log_dir) if K.backend() == 'tensorflow' else Callback()
    gym = Gym(generator=colorizer, critic=critic, combined=combined,
              image_generator=image_generator,
              data_mapper=data_mapper,
              logger=logger,
              models_save_dir=models_save_dir,
              colored_images_save_dir=colored_images_save_dir,
              classifier=classifier)
    gym.train(loss_threshold=loss_threshold,
              eval_interval=eval_interval, epochs=epochs,
              include_target_image=include_target_image)


if __name__ == '__main__':
    fire.Fire(main)

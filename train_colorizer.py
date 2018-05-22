from __future__ import print_function

import os

import fire
import keras
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imsave

from models.colorizer import get_colorizer
from util.colorspace.initialize import get_mapping_with_class_weights
from util.data import ImageGenerator


class Evaluator(Callback):
    def __init__(self, result_save_dir, prediction_to_rgb, data_generator):
        super(Evaluator, self).__init__()
        self.result_save_dir = result_save_dir
        self.prediction_to_rgb = prediction_to_rgb
        self.data_generator = data_generator

        if not os.path.exists(self.result_save_dir):
            os.mkdir(self.result_save_dir)

    def on_epoch_end(self, epoch, logs=None):
        super(Evaluator, self).on_epoch_end(epoch, logs)
        input_images, rgb_images = next(self.data_generator)
        colored_images = self.model.predict(input_images)

        for i, (colored_image, input_image, rgb_image) in enumerate(zip(colored_images, input_images, rgb_images)):
            rgb_prediction = self.prediction_to_rgb(prediction=colored_image, inputs=input_image)
            imsave(name=os.path.join(self.result_save_dir, 'epoch-{}-{}-colored.jpg'.format(epoch, i)), arr=rgb_prediction)
            imsave(name=os.path.join(self.result_save_dir, 'epoch-{}-{}-target.jpg'.format(epoch, i)), arr=rgb_image)


def main(batch_size=32, image_size=224, epochs=100000, steps_per_epoch=100, validation_steps=10, color_space='yuv',
         train_data_dir='/mnt/bolbol/raw-data/train', test_data_dir='/mnt/bolbol/raw-data/validation',
         log_dir='logs', models_save_dir='coloring_models', colored_images_save_dir='colored_images',
         classifier=False, populate_batches=1000, scale_factor=9., weights_file_path=None,
         vgg=False, feature_extractor_model_path=None, train_feature_extractor=False):
    """ Train only colorizer on target images """

    ''' Prepare data generators '''
    train_data_generator = ImageDataGenerator().flow_from_directory(directory=train_data_dir, interpolation='bilinear', target_size=(image_size, image_size), batch_size=batch_size, color_mode='rgb', class_mode=None)
    valid_data_generator = ImageDataGenerator().flow_from_directory(directory=test_data_dir, interpolation='bilinear', target_size=(image_size, image_size), batch_size=batch_size, color_mode='rgb', class_mode=None)
    test_data_generator = ImageDataGenerator().flow_from_directory(directory=test_data_dir, interpolation='bilinear', target_size=(image_size, image_size), batch_size=batch_size, color_mode='rgb', class_mode=None)
    data_mapper, class_weights = get_mapping_with_class_weights(classifier=classifier, color_space=color_space,
                                                                image_generator=train_data_generator,
                                                                image_size=image_size,
                                                                nb_batches=populate_batches, scale_factor=scale_factor,
                                                                weights_file_path=weights_file_path)

    train_data_generator = ImageGenerator(rgb_generator=train_data_generator, input_processing_function=data_mapper.rgb_to_colorizer_input, label_processing_function=data_mapper.rgb_to_colorizer_target)
    valid_data_generator = ImageGenerator(rgb_generator=valid_data_generator, input_processing_function=data_mapper.rgb_to_colorizer_input, label_processing_function=data_mapper.rgb_to_colorizer_target)
    test_data_generator = ImageGenerator(rgb_generator=test_data_generator, input_processing_function=data_mapper.rgb_to_colorizer_input, label_processing_function=lambda x: x)
    ''' Prepare Models '''
    colorizer = get_colorizer(image_size=None, vgg=vgg, feature_extractor_model_path=feature_extractor_model_path,
                              train_feature_extractor=train_feature_extractor,
                              classifier=classifier,
                              classes_per_pixel=class_weights.shape[-1] if classifier else 0)
    colorizer.compile(optimizer=Adam(lr=3e-4), loss='categorical_crossentropy' if classifier else 'mse')

    ''' View summary of the models '''
    print('\n\n\n\nColorizer:'),    colorizer.summary()

    ''' Train the model '''
    logger = keras.callbacks.TensorBoard(log_dir=log_dir) if K.backend() == 'tensorflow' else Callback()
    model_save_path = os.path.join(models_save_dir, 'epoch-{epoch:02d}-loss--{val_loss:.2f}.hdf5')
    colorizer.fit_generator(generator=train_data_generator, steps_per_epoch=steps_per_epoch,
                            validation_data=valid_data_generator, validation_steps=validation_steps,
                            epochs=epochs,
                            class_weight=class_weights,
                            callbacks=[logger,
                                       ModelCheckpoint(filepath=model_save_path),
                                       Evaluator(result_save_dir=colored_images_save_dir,
                                                 prediction_to_rgb=data_mapper.network_prediction_to_rgb,
                                                 data_generator=test_data_generator)])


if __name__ == '__main__':
    fire.Fire(main)

from __future__ import division
from __future__ import print_function

import os

import PIL.Image
import fire
from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine import InputLayer
from keras.layers import Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, img_to_array

from util.colorspace.mapping import get_mapper


def main(batch_size=64, epochs=300, images_per_epoch=5000, validation_images=1024, image_size=224, color_space='yuv',
         train_data_dir='/mnt/bolbol/raw-data/train', valid_data_dir='/mnt/bolbol/raw-data/validation',
         model_save_dir='finetune_models'):
    """ FineTune VGG16 to work on black and white images that are passed as inputs to colorizer """
    data_mapper = get_mapper(color_space=color_space, classifier=False)

    ''' Modify VGG16 to work with greyscale images '''
    vgg = VGG16()
    for layer in vgg.layers:
        layer.trainable = False
    vgg.get_layer(name='block1_conv1').trainable = True
    vgg.get_layer(name='block1_conv2').trainable = True
    vgg.get_layer(name='block2_conv1').trainable = True
    vgg.get_layer(name='block2_conv2').trainable = True

    needed_layers = vgg.layers[2:]
    model = Sequential()
    model.add(InputLayer(input_shape=(image_size, image_size, 1), name='L'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
    for layer in needed_layers:
        model.add(layer)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    ''' Prepare data generators '''
    def preprocess(image):
        image = image.resize((image_size, image_size), PIL.Image.NEAREST)
        image = img_to_array(image)
        return data_mapper.rgb_to_colorizer_input(image)

    generator = ImageDataGenerator(preprocessing_function=preprocess)
    train_generator = generator.flow_from_directory(directory=train_data_dir,
                                                    batch_size=batch_size,
                                                    color_mode='rgb',
                                                    class_mode='categorical')
    valid_generator = generator.flow_from_directory(directory=valid_data_dir,
                                                    batch_size=batch_size,
                                                    color_mode='rgb',
                                                    class_mode='categorical')
    train_generator.image_shape = (image_size, image_size, 1)
    valid_generator.image_shape = (image_size, image_size, 1)
    train_generator.target_size = None
    valid_generator.target_size = None

    # Configure model checkpoints
    model_save_dir = model_save_dir
    model_save_path = os.path.join(model_save_dir, 'finetune-{epoch:02d}-{val_loss:.2f}.hdf5')
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    ''' FineTune VGG '''
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=images_per_epoch // batch_size,
                        epochs=epochs,
                        validation_data=valid_generator,
                        validation_steps=validation_images // batch_size,
                        callbacks=[EarlyStopping(patience=5),
                                   ModelCheckpoint(filepath=model_save_path)])


if __name__ == '__main__':
    fire.Fire(main)

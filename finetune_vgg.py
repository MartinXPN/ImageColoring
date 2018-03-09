from __future__ import division
from __future__ import print_function

import argparse
import os

from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine import InputLayer
from keras.layers import Conv2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from util.data import rgb_to_colorizer_input


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',         default=70,     help='Batch size',                          type=int)
    parser.add_argument('--image_size',         default=224,    help='Batch size',                          type=int)
    parser.add_argument('--epoch_images',       default=5000,   help='Number of images seen in one epoch',  type=int)
    parser.add_argument('--finetune_epochs',    default=300,    help='Number of max epochs for fineTuning', type=int)
    parser.add_argument('--valid_images',       default=1024,   help='Number of images seen during validation', type=int)
    parser.add_argument('--train_data_dir',     default='/mnt/bolbol/raw-data/train',                       type=str)
    parser.add_argument('--valid_data_dir',     default='/mnt/bolbol/raw-data/validation',                  type=str)
    parser.add_argument('--model_save_dir',     default='finetune_models',                                  type=str)
    args = parser.parse_args()

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
    model.add(InputLayer(input_shape=(args.image_size, args.image_size, 1), name='L'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
    for layer in needed_layers:
        model.add(layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    ''' Prepare data generators '''
    generator = ImageDataGenerator(preprocessing_function=rgb_to_colorizer_input)
    train_generator = generator.flow_from_directory(directory=args.train_data_dir,
                                                    target_size=(args.image_size, args.image_size),
                                                    batch_size=args.batch_size,
                                                    color_mode='rgb',
                                                    class_mode='categorical')
    valid_generator = generator.flow_from_directory(directory=args.valid_data_dir,
                                                    target_size=(args.image_size, args.image_size),
                                                    batch_size=args.batch_size,
                                                    color_mode='rgb',
                                                    class_mode='categorical')

    train_generator.image_shape = train_generator.target_size + (1,)
    valid_generator.image_shape  = valid_generator.target_size + (1,)

    # Configure model checkpoints
    model_save_dir = args.model_save_dir
    model_save_path = os.path.join(model_save_dir, 'finetune-{epoch:02d}-{val_loss:.2f}.hdf5')
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    ''' FineTune VGG '''
    model.fit_generator(train_generator,
                        steps_per_epoch=args.epoch_images // args.batch_size,
                        epochs=args.finetune_epochs,
                        validation_data=valid_generator,
                        validation_steps=args.valid_images // args.batch_size,
                        callbacks=[EarlyStopping(patience=5),
                                   ModelCheckpoint(filepath=model_save_path)])

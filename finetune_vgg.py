from __future__ import division
from __future__ import print_function

import argparse

import gc
from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine import InputLayer
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


def preprocess_image(x):
    x -= 128.
    x /= 128
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',         default=70,     help='Batch size',                          type=int)
    parser.add_argument('--image_size',         default=224,    help='Batch size',                          type=int)
    parser.add_argument('--epoch_images',       default=5000,   help='Number of images seen in one epoch',  type=int)
    parser.add_argument('--finetune_epochs',    default=300,    help='Number of max epochs for fineTuning', type=int)
    parser.add_argument('--endtoend_epochs',    default=500,    help='Number of max epochs for end-to-end', type=int)
    parser.add_argument('--valid_images',       default=1024,   help='Number of images seen during validation', type=int)
    parser.add_argument('--train_data_dir',     default='/mnt/bolbol/raw-data/train',                       type=str)
    parser.add_argument('--valid_data_dir',     default='/mnt/bolbol/raw-data/validation',                  type=str)
    args = parser.parse_args()

    vgg = VGG16()
    for layer in vgg.layers:
        layer.trainable = False
    vgg.get_layer(name='block1_conv1').trainable = True
    vgg.get_layer(name='block1_conv2').trainable = True
    vgg.get_layer(name='block2_conv1').trainable = True
    vgg.get_layer(name='block2_conv2').trainable = True

    needed_layers = vgg.layers[2:]
    model = Sequential()
    model.add(InputLayer(input_shape=(args.image_size, args.image_size, 1)))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
    for layer in needed_layers:
        model.add(layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    del vgg, needed_layers[:], needed_layers
    gc.collect()

    generator = ImageDataGenerator(preprocessing_function=preprocess_image)
    train_generator = generator.flow_from_directory(directory=args.train_data_dir,
                                                    target_size=(args.image_size, args.image_size),
                                                    batch_size=args.batch_size,
                                                    color_mode='grayscale')
    valid_generator = generator.flow_from_directory(directory=args.valid_data_dir,
                                                    target_size=(args.image_size, args.image_size),
                                                    batch_size=args.batch_size,
                                                    color_mode='grayscale')

    model.fit_generator(train_generator,
                        steps_per_epoch=args.epoch_images // args.batch_size,
                        epochs=args.finetune_epochs,
                        validation_data=valid_generator,
                        validation_steps=args.valid_images // args.batch_size,
                        callbacks=[EarlyStopping(patience=5),
                                   ModelCheckpoint(filepath='models/finetune-{epoch:02d}-{val_loss:.2f}.hdf5')])

    # Prepare for end-to-end training
    end_to_end_model = Sequential()
    for layer in model.layers:
        if isinstance(layer, MaxPooling2D):     end_to_end_model.add(AveragePooling2D())
        else:                                   end_to_end_model.add(layer)

    for layer in end_to_end_model.layers:
        layer.trainable = True

    del model
    gc.collect()
    end_to_end_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    end_to_end_model.summary()
    end_to_end_model.fit_generator(train_generator,
                                   steps_per_epoch=args.epoch_images // args.batch_size,
                                   epochs=args.endtoend_epochs,
                                   validation_data=valid_generator,
                                   validation_steps=args.valid_images // args.batch_size,
                                   callbacks=[EarlyStopping(patience=5),
                                              ModelCheckpoint(filepath='models/end-to-end-{epoch:02d}-{val_loss:.2f}.hdf5')])

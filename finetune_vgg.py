from __future__ import print_function
from __future__ import division

import argparse
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',     default=70,     help='Batch size',                              type=int)
    parser.add_argument('--image_size',     default=224,    help='Batch size',                              type=int)
    parser.add_argument('--epoch_images',   default=5000,   help='Number of images seen in one epoch',      type=int)
    parser.add_argument('--epochs',         default=500,    help='Number of max epochs',                    type=int)
    parser.add_argument('--valid_images',   default=1024,   help='Number of images seen during validation', type=int)
    parser.add_argument('--train_data_dir', default='/mnt/bolbol/raw-data/train',                           type=str)
    parser.add_argument('--valid_data_dir', default='/mnt/bolbol/raw-data/validation',                      type=str)
    args = parser.parse_args()

    vgg = VGG16()
    for layer in vgg.layers:
        layer.trainable = False
    vgg.get_layer(name='block1_conv1').trainable = True
    vgg.get_layer(name='block1_conv2').trainable = True
    vgg.get_layer(name='block2_conv1').trainable = True
    vgg.get_layer(name='block2_conv2').trainable = True
    vgg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    vgg.summary()

    print('Model layers:')
    for layer in vgg.layers:
        print(layer.name + ':\t', layer.trainable)

    generator = ImageDataGenerator(rescale=1./255)
    train_generator = generator.flow_from_directory(directory=args.train_data_dir,
                                                    target_size=(args.image_size, args.image_size),
                                                    batch_size=args.batch_size)
    valid_generator = generator.flow_from_directory(directory=args.valid_data_dir,
                                                    target_size=(args.image_size, args.image_size),
                                                    batch_size=args.batch_size)

    vgg.fit_generator(train_generator,
                      steps_per_epoch=args.epoch_images // args.batch_size,
                      epochs=args.epochs,
                      validation_data=valid_generator,
                      validation_steps=args.valid_images // args.batch_size,
                      callbacks=[TensorBoard(),
                                 EarlyStopping(patience=2),
                                 ModelCheckpoint(filepath='models/checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5')])

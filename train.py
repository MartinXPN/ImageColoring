import argparse

from keras.preprocessing.image import ImageDataGenerator

import generators
from models.colorizer import Colorizer
from models.critic import Critic
from models.gan import CombinedGan


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',         default=70,     help='Batch size',                          type=int)
    parser.add_argument('--image_size',         default=224,    help='Batch size',                          type=int)
    parser.add_argument('--epoch_images',       default=5000,   help='Number of images seen in one epoch',  type=int)
    parser.add_argument('--train_data_dir',     default='/mnt/bolbol/raw-data/train',                       type=str)
    parser.add_argument('--valid_data_dir',     default='/mnt/bolbol/raw-data/validation',                  type=str)
    parser.add_argument('--feature_extractor_model_path',
                        default='/Users/martin/Desktop/finetune-40-2.08-no-top.hdf5',
                        help='Path to VGG/Feature extractor model or weights')
    args = parser.parse_args()

    colorizer = Colorizer(feature_extractor_model_path=args.feature_extractor_model_path,
                          input_shape=(args.image_size, args.image_size, 1))
    critic = Critic(input_shape=(args.image_size, args.image_size, 3))
    combined = CombinedGan(generator=colorizer, critic=critic, input_shape=(args.image_size, args.image_size, 1))

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
                                                        color_mode='rgb',
                                                        label=-1)
    combined_generator = generators.ImageDataGenerator(directory=args.train_data_dir,
                                                       image_data_generator=generator,
                                                       target_size=(args.image_size, args.image_size),
                                                       batch_size=args.batch_size,
                                                       color_mode='rgb',
                                                       class_mode='input',
                                                       label=-1)

    combined_generator.next()

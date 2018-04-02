from __future__ import division

from keras import Input
from keras.engine import Model
from keras.layers import Conv2D, MaxPooling2D, Concatenate, UpSampling2D, Activation, PReLU


class Colorizer(Model):
    def __init__(self, input_shape=(None, None, 1),
                 feature_extractor_model_path=None, train_feature_extractor=False,
                 inputs=None, outputs=None, name='Colorizer'):
        if inputs and outputs:
            super(Colorizer, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        gray = Input(shape=input_shape, name='gray')

        ''' Construct VGG16 architecture '''
        x = gray
        concat_layers = []
        for filters, num_layers in zip([64, 128, 256, 512, 512], [2, 2, 3, 3, 3]):
            concat_layers.append(x)
            for i in range(num_layers):
                x = Conv2D(filters=filters, kernel_size=3, activation=None, padding='same')(x)
                x = Activation('relu')(x) if feature_extractor_model_path else PReLU()(x)
            x = MaxPooling2D(pool_size=2, strides=2)(x) if feature_extractor_model_path else \
                Conv2D(filters=filters // 4, kernel_size=3, strides=2, padding='same')(x)

        ''' Load pre-trained VGG16 weights and fix them '''
        if feature_extractor_model_path is not None:
            feature_extractor = Model(gray, x, name='vgg16')
            feature_extractor.load_weights(feature_extractor_model_path)
            feature_extractor.trainable = train_feature_extractor
            for l in feature_extractor.layers:
                l.trainable = train_feature_extractor

        ''' Get the output of feature extractor and add up-sampling layers on top of the features '''
        concat_layers = concat_layers[::-1]
        for filters, concat_layer, num_layers in zip([256, 256, 128, 64, 32], concat_layers, [2, 2, 2, 2, 2]):
            for i in range(num_layers):
                x = Conv2D(filters, kernel_size=(3, 3), activation=None, padding='same')(x)
                x = PReLU()(x)
            # x = SubpixelUpSampling(filters=filters, kernel_size=3, ratio=2, padding='same')(x)
            x = UpSampling2D(size=(2, 2))(x)
            x = Concatenate()([concat_layer, x])

        ''' Post-processing after pix2pix-like connections '''
        for filters in [64, 64, 32]:
            x = Conv2D(filters, kernel_size=(3, 3), activation=None, padding='same')(x)
            x = PReLU()(x)
        colored = Conv2D(2, kernel_size=(3, 3), activation='tanh', padding='same', name='out')(x)
        super(Colorizer, self).__init__(inputs=gray, outputs=colored, name=name)

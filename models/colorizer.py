from __future__ import division

from keras import Input
from keras.engine import Model
from keras.layers import Conv2D, concatenate, MaxPooling2D, ELU

from layers.upsampling import SubpixelUpSampling


class Colorizer(Model):
    def __init__(self, feature_extractor_model_path=None, input_shape=(None, None, 1),
                 inputs=None, outputs=None, name='Colorizer'):
        if inputs and outputs:
            super(Colorizer, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        L = Input(shape=input_shape, name='L')

        ''' Construct VGG16 architecture '''
        x = L
        concat_layers = []
        for filters, num_layers in zip([64, 128, 256, 512, 512], [2, 2, 3, 3, 3]):
            concat_layers.append(x)
            for i in range(num_layers):
                x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        ''' Load pre-trained VGG16 weights and fix them '''
        feature_extractor = Model(L, x, name='vgg16')
        feature_extractor.load_weights(feature_extractor_model_path)
        feature_extractor.trainable = False
        for l in feature_extractor.layers:
            l.trainable = False

        ''' Get the output of feature extractor and add up-sampling layers on top of the features '''
        concat_layers = concat_layers[::-1]
        for filters, concat_layer, num_layers in zip([256, 256, 128, 64, 32], concat_layers, [2, 2, 2, 2, 2]):
            for i in range(num_layers):
                x = Conv2D(filters, kernel_size=(3, 3), activation=None, padding='same')(x)
                x = ELU()(x)
            x = SubpixelUpSampling(filters=filters, kernel_size=3, ratio=2, padding='same')(x)
            x = concatenate(inputs=[concat_layer, x])

        ''' Post-processing after pix2pix-like connections '''
        for filters in [64, 64, 32]:
            x = Conv2D(filters, kernel_size=(3, 3), activation=None, padding='same')(x)
            x = ELU()(x)
        ab = Conv2D(2, kernel_size=(3, 3), activation='tanh', padding='same', name='ab')(x)
        super(Colorizer, self).__init__(inputs=L, outputs=ab, name=name)

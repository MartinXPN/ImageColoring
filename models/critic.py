from keras import Input
from keras.engine import Model
from keras.initializers import RandomNormal
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Dropout, PReLU

from util.clipweights import WeightClip

weight_init = RandomNormal(mean=0., stddev=0.02)


def Conv2DBatchNormPReLU(input_layer,
                         filters, kernel_size, strides=(1, 1), padding='valid'):
    res = input_layer
    res = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                 activation=None, padding=padding,
                 kernel_initializer=weight_init,
                 kernel_constraint=WeightClip(-0.01, 0.01), bias_constraint=WeightClip(-0.01, 0.01))(res)
    res = BatchNormalization()(res)
    res = PReLU()(res)
    return res


class Critic(Model):
    def __init__(self, input_shape=(224, 224, 3),
                 inputs=None, outputs=None, name='Critic'):
        if inputs and outputs:
            super(Critic, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        ''' VGG-like conv filters '''
        input_image = Input(shape=input_shape)
        x = input_image
        for filters, kernel_size, strides in zip([32, 64, 128, 256, 256], [5, 5, 5, 5, 3], [2, 2, 2, 2, 1]):
            x = Conv2DBatchNormPReLU(x, filters=filters, kernel_size=kernel_size, strides=strides)
            x = Dropout(rate=0.1)(x)

        ''' Fully connected layers '''
        x = Flatten()(x)
        for units in [128]:
            x = Dense(units=units, activation=None,
                      kernel_initializer=weight_init,
                      kernel_constraint=WeightClip(-0.01, 0.01), bias_constraint=WeightClip(-0.01, 0.01))(x)
            x = BatchNormalization()(x)
            x = PReLU()(x)
            x = Dropout(rate=0.1)(x)

        out = Dense(1, activation=None)(x)
        super(Critic, self).__init__(inputs=input_image, outputs=out, name=name)

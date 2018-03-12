from keras import Input
from keras.engine import Model
from keras.initializers import RandomNormal
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, Dense, Flatten, Dropout

from util.clipweights import WeightClip


weight_init = RandomNormal(mean=0., stddev=0.02)


def Conv2DBatchNormalizationLeakyReLU(input_layer,
                                      filters, kernel_size, strides=(1, 1), padding='valid',
                                      alpha=0.3):
    res = input_layer
    res = Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 strides=strides,
                 activation=None,
                 padding=padding,
                 kernel_initializer=weight_init,
                 kernel_constraint=WeightClip(-0.01, 0.01),
                 bias_constraint=WeightClip(-0.01, 0.01))(res)
    res = BatchNormalization()(res)
    res = LeakyReLU(alpha=alpha)(res)
    return res


class Critic(Model):
    def __init__(self, input_shape=(224, 224, 3),
                 inputs=None, outputs=None, name='Critic'):
        if inputs and outputs:
            super(Critic, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        ''' VGG-like conv filters '''
        lab = Input(shape=input_shape)
        x = lab
        for filters in [64, 256, 512, 64]:
            x = Conv2DBatchNormalizationLeakyReLU(x, filters=filters, kernel_size=(3, 3), strides=(1, 1))
            x = Conv2DBatchNormalizationLeakyReLU(x, filters=filters, kernel_size=(3, 3), strides=(1, 1))
            x = Conv2DBatchNormalizationLeakyReLU(x, filters=filters, kernel_size=(5, 5), strides=(2, 2))
            x = Dropout(rate=0.3)(x)

        ''' Fully connected layers '''
        x = Flatten()(x)
        for units in [1024, 128]:
            x = Dense(units=units, activation=None,
                      kernel_initializer=weight_init, kernel_constraint=WeightClip(-0.01, 0.01),
                      bias_constraint=WeightClip(-0.01, 0.01))(x)
            x = LeakyReLU()(x)
            x = Dropout(rate=0.3)(x)

        out = Dense(1, activation='linear')(x)
        super(Critic, self).__init__(inputs=lab, outputs=out, name=name)

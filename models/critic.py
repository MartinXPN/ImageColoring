from keras import Input
from keras.engine import Model
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, Dense, Flatten, Dropout


def Conv2DBatchNormalizationLeakyReLU(input_layer,
                                      filters, kernel_size, strides=(1, 1), padding='valid',
                                      alpha=0.3):
    res = input_layer
    res = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation=None, padding=padding)(res)
    res = BatchNormalization()(res)
    res = LeakyReLU(alpha=alpha)(res)
    return res


class Critic(Model):
    def __init__(self, input_shape=(224, 224, 3),
                 inputs=None, outputs=None, name='Critic'):
        if inputs and outputs:
            super(Critic, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        rgb = Input(shape=input_shape)
        x = rgb
        for filters in [64, 256, 512, 64]:
            x = Conv2DBatchNormalizationLeakyReLU(x, filters=filters, kernel_size=(3, 3), strides=(1, 1))
            x = Conv2DBatchNormalizationLeakyReLU(x, filters=filters, kernel_size=(3, 3), strides=(1, 1))
            x = Conv2DBatchNormalizationLeakyReLU(x, filters=filters, kernel_size=(5, 5), strides=(2, 2))

        x = Flatten()(x)
        x = Dropout(rate=0.3)(x)

        x = Dense(1024, activation=None)(x)
        x = LeakyReLU()(x)
        x = Dropout(rate=0.3)(x)

        x = Dense(1024, activation=None)(x)
        x = LeakyReLU()(x)
        x = Dropout(rate=0.3)(x)
        out = Dense(1, activation='linear')(x)
        super(Critic, self).__init__(inputs=rgb, outputs=out, name=name)

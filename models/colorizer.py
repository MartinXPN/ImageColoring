from keras import Input
from keras.engine import Model
from keras.layers import Conv2D, UpSampling2D, concatenate, MaxPooling2D


class Colorizer(Model):
    def __init__(self, feature_extractor_model_path=None, input_shape=(None, None, 1),
                 inputs=None, outputs=None, name='Colorizer'):
        if inputs and outputs:
            super(Colorizer, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        L = Input(shape=input_shape, name='L')

        # Encoder 1
        e1 = Conv2D(64, (3, 3), activation='relu', padding='same')(L)
        e1 = Conv2D(64, (3, 3), activation='relu', padding='same')(e1)
        e1 = MaxPooling2D((2, 2), strides=(2, 2))(e1)

        # Encoder 2
        e2 = Conv2D(128, (3, 3), activation='relu', padding='same')(e1)
        e2 = Conv2D(128, (3, 3), activation='relu', padding='same')(e2)
        e2 = MaxPooling2D((2, 2), strides=(2, 2))(e2)

        # Encoder 3
        e3 = Conv2D(256, (3, 3), activation='relu', padding='same')(e2)
        e3 = Conv2D(256, (3, 3), activation='relu', padding='same')(e3)
        e3 = Conv2D(256, (3, 3), activation='relu', padding='same')(e3)
        e3 = MaxPooling2D((2, 2), strides=(2, 2))(e3)

        # Encoder 4
        e4 = Conv2D(512, (3, 3), activation='relu', padding='same')(e3)
        e4 = Conv2D(512, (3, 3), activation='relu', padding='same')(e4)
        e4 = Conv2D(512, (3, 3), activation='relu', padding='same')(e4)
        e4 = MaxPooling2D((2, 2), strides=(2, 2))(e4)

        # Encoder 5
        e5 = Conv2D(512, (3, 3), activation='relu', padding='same')(e4)
        e5 = Conv2D(512, (3, 3), activation='relu', padding='same')(e5)
        e5 = Conv2D(512, (3, 3), activation='relu', padding='same')(e5)
        e5 = MaxPooling2D((2, 2), strides=(2, 2))(e5)

        feature_extractor = Model(L, e5, name='vgg16')
        feature_extractor.load_weights(feature_extractor_model_path)
        feature_extractor.trainable = False

        # Get the output of feature extractor and add up-sampling layers on top of the features
        x = feature_extractor(L)
        for filters, concat_layer in zip([512, 256, 128, 64, 32], [e4, e3, e2, e1, L]):
            x = Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
            x = Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
            x = Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
            x = UpSampling2D(size=(2, 2))(x)
            x = concatenate(inputs=[concat_layer, x])

        x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(x)
        ab = Conv2D(2, kernel_size=(3, 3), activation='tanh', padding='same', name='ab')(x)

        super(Colorizer, self).__init__(inputs=L, outputs=ab, name=name)

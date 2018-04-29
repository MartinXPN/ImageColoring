from __future__ import division

from keras import Input
from keras.engine import Model
from keras.layers import Conv2D, MaxPooling2D, Concatenate, UpSampling2D, PReLU, Softmax
from keras.models import load_model


class Colorizer(Model):
    def __init__(self, input_shape=(None, None, 1),
                 inputs=None, outputs=None, name='Colorizer'):
        if inputs and outputs:
            super(Colorizer, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        gray = Input(shape=input_shape, name='gray')
        x, concat_layers = self.construct_encoding_trunk(gray)
        x = self.construct_decoding_trunk(x, concat_layers=concat_layers)
        colored = self.construct_post_process_trunk(x)

        super(Colorizer, self).__init__(inputs=gray, outputs=colored, name=name)

    def construct_encoding_trunk(self, inputs):
        x = inputs
        concat_layers = []
        for filters, num_layers in zip([64, 128, 256, 512, 512], [2, 2, 3, 3, 3]):
            concat_layers.append(x)
            for i in range(num_layers):
                x = Conv2D(filters=filters, kernel_size=3, activation=None, padding='same')(x)
                x = PReLU()(x)
            x = Conv2D(filters=filters // 4, kernel_size=3, strides=2, padding='same')(x)
        return x, concat_layers[::-1]

    @staticmethod
    def construct_decoding_trunk(x, concat_layers):
        for filters, concat_layer, num_layers in zip([256, 256, 128, 64, 32], concat_layers, [2, 2, 2, 2, 2]):
            for i in range(num_layers):
                x = Conv2D(filters, kernel_size=3, activation=None, padding='same')(x)
                x = PReLU()(x)
            # x = SubpixelUpSampling(filters=filters, kernel_size=3, ratio=2, padding='same')(x)
            # x = Conv2DTranspose(filters=filters, kernel_size=3)(x)
            x = UpSampling2D(size=2)(x)
            x = Concatenate()([concat_layer, x])
        return x

    @staticmethod
    def construct_post_process_trunk(x):
        for filters in [64, 64, 32]:
            x = Conv2D(filters, kernel_size=3, activation=None, padding='same')(x)
            x = PReLU()(x)
        x = Conv2D(2, kernel_size=3, activation='tanh', padding='same', name='out')(x)
        return x


class VGGColorizer(Colorizer):
    def __init__(self, input_shape=(None, None, 1),
                 feature_extractor_model_path=None, train_feature_extractor=False,
                 inputs=None, outputs=None, name='Colorizer'):
        self.feature_extractor_model_path = feature_extractor_model_path
        self.train_feature_extractor = train_feature_extractor
        super(VGGColorizer, self).__init__(input_shape=input_shape,
                                           inputs=inputs, outputs=outputs, name=name)

    def construct_encoding_trunk(self, inputs):
        x = inputs
        concat_layers = []
        for filters, num_layers in zip([64, 128, 256, 512, 512], [2, 2, 3, 3, 3]):
            concat_layers.append(x)
            for i in range(num_layers):
                x = Conv2D(filters=filters, kernel_size=3, activation='relu', padding='same')(x)
            x = MaxPooling2D(pool_size=2, strides=2)(x)

        ''' Load pre-trained VGG16 weights and fix them '''
        if self.feature_extractor_model_path is not None:
            feature_extractor = Model(inputs, x, name='vgg16')
            feature_extractor.load_weights(self.feature_extractor_model_path)
            feature_extractor.trainable = self.train_feature_extractor
            for l in feature_extractor.layers:
                l.trainable = self.train_feature_extractor

        return x, concat_layers[::-1]


class VGGClassificationColorizer(VGGColorizer):
    def __init__(self, input_shape=(None, None, 1), classes_per_pixel=100,
                 feature_extractor_model_path=None, train_feature_extractor=False,
                 inputs=None, outputs=None, name='Colorizer'):
        self.classes_per_pixel = classes_per_pixel
        super(VGGClassificationColorizer, self).__init__(input_shape=input_shape,
                                                         feature_extractor_model_path=feature_extractor_model_path,
                                                         train_feature_extractor=train_feature_extractor,
                                                         inputs=inputs, outputs=outputs, name=name)

    def construct_post_process_trunk(self, x):
        concat = x
        for filters in [64, 64]:
            x = Conv2D(filters, kernel_size=3, activation=None, padding='same')(concat)
            x = PReLU()(x)
            concat = x  # Concatenate()([concat, x])

        x = concat
        x = Conv2D(filters=self.classes_per_pixel, kernel_size=3, activation=None, padding='same')(x)
        x = Softmax(axis=-1)(x)
        return x


def get_colorizer(colorizer_model_path=None,
                  image_size=224, vgg=False, feature_extractor_model_path=None, train_feature_extractor=False,
                  classifier=False, classes_per_pixel=300):
    if colorizer_model_path:
        return load_model(filepath=colorizer_model_path, compile=False,
                          custom_objects={'Colorizer': Colorizer,
                                          'VGGColorizer': VGGColorizer,
                                          'VGGClassificationClassifier': VGGClassificationColorizer})
    elif classifier and vgg:
        return VGGClassificationColorizer(input_shape=(image_size, image_size, 1), classes_per_pixel=classes_per_pixel,
                                          feature_extractor_model_path=feature_extractor_model_path,
                                          train_feature_extractor=train_feature_extractor)
    elif vgg:
        return VGGColorizer(input_shape=(image_size, image_size, 1),
                            feature_extractor_model_path=feature_extractor_model_path,
                            train_feature_extractor=train_feature_extractor)
    else:
        return Colorizer(input_shape=(image_size, image_size, 1))

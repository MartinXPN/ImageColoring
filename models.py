from keras.activations import relu
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, concatenate, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils.data_utils import get_file

VGG_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def create_colorizer(input_shape=(None, None, 3)):

    inputs = Input(shape=input_shape)
    
    # Encoder 1
    e11 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    e12 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(e11)
    e13 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(e12)

    # Encoder 2
    e21 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(e13)
    e22 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(e21)
    e23 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(e22)

    # Encoder 3
    e31 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(e23)
    e32 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(e31)
    e33 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(e32)
    e34 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(e33)

    # Encoder 4
    e41 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(e34)
    e42 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(e41)
    e43 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(e42)
    e44 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(e43)

    # Encoder 5
    e51 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(e44)
    e52 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(e51)
    e53 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(e52)
    e54 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(e53)
    
    # *********************************************************************************** #
    # Now lets decode the representation                                                  #
    # Using residual connections (in this case concatenate)                               #
    # *********************************************************************************** #

    # Decoder 5
    d51 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_deconv1')(e54)
    d52 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_deconv2')(d51)
    d53 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_deconv3')(d52)
    d54 = UpSampling2D((2, 2), name='block5_upsample')(d53)
    
    # Decoder 4
    merged = concatenate([d54, e44])
    d41 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_deconv1')(merged)
    d42 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_deconv2')(d41)
    d43 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_deconv3')(d42)
    d44 = UpSampling2D((2, 2), name='block4_upsample')(d43)
    
    # Decoder 3
    merged = concatenate([d44, e34])
    d31 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_deconv1')(merged)
    d32 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_deconv2')(d31)
    d33 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_deconv3')(d32)
    d34 = UpSampling2D((2, 2), name='block3_upsample')(d33)
    
    # Decoder 2
    merged = concatenate([d34, e23])
    d21 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_deconv1')(merged)
    d22 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_deconv2')(d21)
    d23 = UpSampling2D((2, 2), name='block2_upsample')(d22)
    
    # Decoder 1
    merged = concatenate([d23, e13])
    d11 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_deconv1')(merged)
    d12 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_deconv2')(d11)
    d13 = UpSampling2D((2, 2), name='block1_upsample')(d12)

    d01 = Conv2D(16, (3, 3), activation='relu',    padding='same', name='out1')(d13)
    out = Conv2D(3, (3, 3), activation=None, padding='same', name='out2')(d01)
    out = Activation(lambda x: relu(x, max_value=255.))(out)

    # Create model.
    vgg16 = Model(inputs, e54, name='vgg16')
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', VGG_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    
    # Load weights of vgg16 and fix them (set non-trainable)
    vgg16.load_weights(weights_path)
    for l in vgg16.layers:
        l.trainable = False
    
    model = Model(inputs, out, name='colorizer')
    return model, vgg16


def create_discriminator(input_image_shape=(224, 224, 3), initial_image_shape=(224, 224, 3)):

    assert input_image_shape == initial_image_shape

    def Conv2DBatchNormalizationLeakyReLU(input_layer,
                                          filters,
                                          kernel_size,
                                          strides=(1, 1),
                                          padding='valid',
                                          alpha=0.3):
        res = input_layer
        res = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation=None, padding=padding)(res)
        res = BatchNormalization()(res)
        res = LeakyReLU(alpha=alpha)(res)
        return res


    def create_trunk(input_image_shape):
        inputs = Input(shape=input_image_shape)

        # Block 1
        res = Conv2DBatchNormalizationLeakyReLU(inputs, filters=64, kernel_size=(3, 3), strides=(1, 1))
        res = Conv2DBatchNormalizationLeakyReLU(res, filters=64, kernel_size=(3, 3), strides=(2, 2))

        # Block 2
        res = Conv2DBatchNormalizationLeakyReLU(res, filters=128, kernel_size=(3, 3), strides=(1, 1))
        res = Conv2DBatchNormalizationLeakyReLU(res, filters=128, kernel_size=(3, 3), strides=(2, 2))

        # Block 3
        res = Conv2DBatchNormalizationLeakyReLU(res, filters=256, kernel_size=(3, 3), strides=(1, 1))
        res = Conv2DBatchNormalizationLeakyReLU(res, filters=256, kernel_size=(3, 3), strides=(1, 1))
        res = Conv2DBatchNormalizationLeakyReLU(res, filters=256, kernel_size=(3, 3), strides=(1, 1))

        # Block 4
        res = Conv2DBatchNormalizationLeakyReLU(res, filters=512, kernel_size=(3, 3), strides=(1, 1))
        res = Conv2DBatchNormalizationLeakyReLU(res, filters=512, kernel_size=(3, 3), strides=(1, 1))
        res = Conv2DBatchNormalizationLeakyReLU(res, filters=512, kernel_size=(3, 3), strides=(1, 1))

        # Block 5
        res = Conv2DBatchNormalizationLeakyReLU(res, filters=256, kernel_size=(3, 3), strides=(1, 1))
        res = Conv2DBatchNormalizationLeakyReLU(res, filters=256, kernel_size=(3, 3), strides=(1, 1))
        res = Conv2DBatchNormalizationLeakyReLU(res, filters=256, kernel_size=(3, 3), strides=(1, 1))

        # Block 6
        res = Conv2DBatchNormalizationLeakyReLU(res, filters=128, kernel_size=(3, 3), strides=(2, 2))
        res = Conv2DBatchNormalizationLeakyReLU(res, filters=128, kernel_size=(3, 3), strides=(1, 1))
        return inputs, res

    # Create trunks separately for real/fake input image and for initial black-white image
    # To extract features from both and then perform a prediction
    input_image_trunk_input, input_image_trunk_output = create_trunk(input_image_shape)
    initial_image_trunk_input, initial_image_trunk_output = create_trunk(initial_image_shape)

    # Merge Trunks to make a decision
    merged_trunks = concatenate([input_image_trunk_output, initial_image_trunk_output])
    merged_trunks = Conv2DBatchNormalizationLeakyReLU(merged_trunks, filters=64, kernel_size=(3, 3), strides=(1, 1))
    merged_trunks = Conv2DBatchNormalizationLeakyReLU(merged_trunks, filters=64, kernel_size=(3, 3), strides=(1, 1))
    merged_trunks = Conv2DBatchNormalizationLeakyReLU(merged_trunks, filters=32, kernel_size=(3, 3), strides=(1, 1))
    merged_trunks = Conv2DBatchNormalizationLeakyReLU(merged_trunks, filters=16, kernel_size=(3, 3), strides=(1, 1))

    x = Flatten()(merged_trunks)
    x = Dense(1024, activation=None)(x)
    x = LeakyReLU()(x)
    out = Dense(1, activation='sigmoid')(x)
    
    model = Model([input_image_trunk_input, initial_image_trunk_input], out, name='discriminator')
    return model


def create_GAN(generator, discriminator):
    generator_trainable_layers     = [layer for layer in generator.layers if layer.trainable]
    discriminator_trainable_layers = [layer for layer in discriminator.layers if layer.trainable]

    network_input = Input(shape=(224, 224, 3))
    generator_output = generator(network_input)
    network_output = discriminator([generator_output, network_input])

    GAN = Model(network_input, outputs=[network_output, generator_output])
    return GAN, generator_trainable_layers, discriminator_trainable_layers

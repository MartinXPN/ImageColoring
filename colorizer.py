VGG_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def create_colorizer(input_shape=(None, None, 3)):
    
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, Deconvolution2D, UpSampling2D, concatenate
    from keras.layers.core import Lambda
    from keras.utils.data_utils import get_file


    # Determine proper input shape
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
    d51 = Deconvolution2D(512, (3, 3), activation='relu', padding='same', name='block5_deconv1')(e54)
    d52 = Deconvolution2D(512, (3, 3), activation='relu', padding='same', name='block5_deconv2')(d51)
    d53 = Deconvolution2D(512, (3, 3), activation='relu', padding='same', name='block5_deconv3')(d52)
    d54 = UpSampling2D((2, 2), name='block5_upsample')(d53)
    
    # Decoder 4
    merged = concatenate([d54, e44])
    d41 = Deconvolution2D(256, (3, 3), activation='relu', padding='same', name='block4_deconv1')(merged)
    d42 = Deconvolution2D(256, (3, 3), activation='relu', padding='same', name='block4_deconv2')(d41)
    d43 = Deconvolution2D(256, (3, 3), activation='relu', padding='same', name='block4_deconv3')(d42)
    d44 = UpSampling2D((2, 2), name='block4_upsample')(d43)
    
    # Decoder 3
    merged = concatenate([d44, e34])
    d31 = Deconvolution2D(128, (3, 3), activation='relu', padding='same', name='block3_deconv1')(merged)
    d32 = Deconvolution2D(128, (3, 3), activation='relu', padding='same', name='block3_deconv2')(d31)
    d33 = Deconvolution2D(128, (3, 3), activation='relu', padding='same', name='block3_deconv3')(d32)
    d34 = UpSampling2D((2, 2), name='block3_upsample')(d33)
    
    # Decoder 2
    merged = concatenate([d34, e23])
    d21 = Deconvolution2D(64, (3, 3), activation='relu', padding='same', name='block2_deconv1')(merged)
    d22 = Deconvolution2D(64, (3, 3), activation='relu', padding='same', name='block2_deconv2')(d21)
    d23 = UpSampling2D((2, 2), name='block2_upsample')(d22)
    
    # Decoder 1
    merged = concatenate([d23, e13])
    d11 = Deconvolution2D(32, (3, 3), activation='relu', padding='valid', name='block1_deconv1')(merged)
    d12 = Deconvolution2D(32, (4, 4), strides=2, activation='relu', padding='valid', name='block1_deconv2')(d11)
    
    d01 = Conv2D(16, (3, 3), activation='relu',    padding='valid', name='out1')(d12)
    d02 = Conv2D(2, (5, 5), activation='tanh', padding='valid', name='out2')(d01)
    out = Lambda(lambda x: x * 128., output_shape=lambda x:x) (d02) # Map sigmoid to [-128; 128]


    
    # Create model.
    vgg16 = Model(inputs, e54, name='vgg16')
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', VGG_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    
    # Load weights of vgg16 and fix them (set non-trainable)
    vgg16.load_weights(weights_path)
    for l in vgg16.layers:
        l.trainable = False
    
    model = Model(inputs, out, name='colorizer')
    return model, vgg16
def create_discriminator(input_shape=(224, 224, 3)):
    
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
    from keras.layers.advanced_activations import LeakyReLU
    from keras.layers.normalization import BatchNormalization

    
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = Conv2D(64, (3, 3), activation=None, padding='valid')(inputs)
    x = LeakyReLU()(x)
    
    x = Conv2D(64, (3, 3), strides=(2, 2), activation=None, padding='valid')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    
    # Block 2
    x = Conv2D(128, (3, 3), activation=None, padding='valid')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(128, (3, 3), strides=(2, 2), activation=None, padding='valid')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    
    # Block 3
    x = Conv2D(256, (3, 3), activation=None, padding='valid')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(256, (3, 3), activation=None, padding='valid')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(256, (3, 3), strides=(2, 2), activation=None, padding='valid')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    
    # Block 4
    x = Conv2D(512, (3, 3), activation=None, padding='valid')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(512, (3, 3), activation=None, padding='valid')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(512, (3, 3), activation=None, padding='valid')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    
    # Block 5
    x = Conv2D(512, (3, 3), activation=None, padding='valid')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(256, (3, 3), activation=None, padding='valid')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(256, (3, 3), activation=None, padding='valid')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(128, (3, 3), activation=None, padding='valid')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(128, (3, 3), activation=None, padding='valid')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(64, (3, 3), activation=None, padding='valid')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    
    
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation=None, name='fc1')(x)
    x = LeakyReLU()(x)
    out = Dense(1, activation='sigmoid', name='out')(x)
    
    model = Model(inputs, out, name='discriminator')
    return model
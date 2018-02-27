from keras import Input
from keras.engine import Model


class CombinedGan(Model):
    def __init__(self, generator=None, critic=None, input_shape=(224, 224, 1),
                 inputs=None, outputs=None, name='Combined'):
        if inputs and outputs:
            super(CombinedGan, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        greyscale_input = Input(shape=input_shape)
        colorized_fake_image = generator(greyscale_input)

        # we only want to be able to train generator/colorizer for the combined model
        critic.trainable = False
        critic_output = critic(colorized_fake_image)

        super(CombinedGan, self).__init__(inputs=greyscale_input,                         # Only one input - grey image
                                          outputs=[critic_output, colorized_fake_image],  # colorized image for L1 loss
                                          name=name)

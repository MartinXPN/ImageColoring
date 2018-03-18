from keras import Input
from keras.engine import Model
from keras.layers import Concatenate


class CombinedGan(Model):
    def __init__(self, generator=None, critic=None, input_shape=(224, 224, 1),
                 inputs=None, outputs=None, name='Combined'):
        if inputs and outputs:
            super(CombinedGan, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        gray = Input(shape=input_shape)
        colorizer_output = generator(gray)
        critic_input = Concatenate()([gray, colorizer_output])

        # we only want to be able to train generator/colorizer for the combined model
        critic.trainable = False
        critic_output = critic(critic_input)

        super(CombinedGan, self).__init__(inputs=gray,                                  # Only one input - grey image
                                          outputs=[critic_output, colorizer_output],    # colorized output for L1 loss
                                          name=name)

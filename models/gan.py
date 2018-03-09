from keras import Input
from keras.engine import Model
from keras.layers import Concatenate


class CombinedGan(Model):
    def __init__(self, generator=None, critic=None, input_shape=(224, 224, 1),
                 inputs=None, outputs=None, name='Combined'):
        if inputs and outputs:
            super(CombinedGan, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        L = Input(shape=input_shape)
        ab = generator(L)
        Lab = Concatenate()([L, ab])

        # we only want to be able to train generator/colorizer for the combined model
        critic.trainable = False
        critic_output = critic(Lab)

        super(CombinedGan, self).__init__(inputs=L,                     # Only one input - grey image
                                          outputs=[critic_output, ab],  # colorized output for L1 loss
                                          name=name)

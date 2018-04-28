from keras import Input
from keras import backend as K
from keras.engine import Model
from keras.layers import Concatenate, Lambda, Embedding, TimeDistributed


class CombinedGan(Model):
    def __init__(self, generator=None, critic=None, input_shape=(224, 224, 1), include_colorizer_output=True,
                 classifier=False, class_to_color=None,
                 inputs=None, outputs=None, name='Combined'):
        if inputs and outputs:
            super(CombinedGan, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        gray = Input(shape=input_shape)
        colorizer_output = generator(gray)
        if classifier:
            colorizer_output = Lambda(lambda x: K.argmax(x))(colorizer_output)
            colorizer_output = Lambda(lambda x: K.expand_dims(x))(colorizer_output)
            colorizer_output = TimeDistributed(TimeDistributed(Embedding(input_dim=class_to_color.shape[0],
                                                                         output_dim=class_to_color.shape[1],
                                                                         weights=[class_to_color],
                                                                         trainable=False)))(colorizer_output)
            print(K.int_shape(colorizer_output))
            colorizer_output = Lambda(lambda x: x / 128.)(colorizer_output)
        critic_input = Concatenate()([gray, colorizer_output])

        # we only want to be able to train generator/colorizer for the combined model
        critic.trainable = False
        critic_output = critic(critic_input)

        # Include colorizer output as one of the outputs or not
        outputs = [critic_output, colorizer_output] if include_colorizer_output else critic_output
        super(CombinedGan, self).__init__(inputs=gray,
                                          outputs=outputs,
                                          name=name)

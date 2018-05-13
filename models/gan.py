from keras import Input
from keras import backend as K
from keras.engine import Model
from keras.layers import Concatenate, Lambda


class CombinedGan(Model):
    def __init__(self, generator=None, critic=None, input_shape=(224, 224, 1), include_colorizer_output=True,
                 inputs=None, outputs=None, name='Combined'):
        if inputs and outputs:
            super(CombinedGan, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return

        gray = Input(shape=input_shape)
        colorizer_output = generator(gray)
        critic_input = self.colorizer_output_to_critic_input(gray, colorizer_output)

        # we only want to be able to train generator/colorizer for the combined model
        critic.trainable = False
        critic_output = critic(critic_input)

        # Include colorizer output as one of the outputs or not
        outputs = [critic_output, colorizer_output] if include_colorizer_output else critic_output
        super(CombinedGan, self).__init__(inputs=gray,
                                          outputs=outputs,
                                          name=name)

    def colorizer_output_to_critic_input(self, gray, colorizer_output):
        return Concatenate()([gray, colorizer_output])


class ClassifierCombinedGan(CombinedGan):
    def __init__(self, class_to_color=None, **kwargs):
        self.class_to_color = K.variable(class_to_color)
        super(ClassifierCombinedGan, self).__init__(**kwargs)

    def colorizer_output_to_critic_input(self, gray, colorizer_output):
        colorizer_output = Lambda(lambda x: K.dot(x, self.class_to_color))(colorizer_output)
        return Concatenate()([gray, colorizer_output])


def get_combined_gan(classifier, class_to_color=None,
                     generator=None, critic=None,
                     input_shape=(224, 224, 1),  include_colorizer_output=True):
    if classifier:
        return ClassifierCombinedGan(generator=generator, critic=critic,
                                     input_shape=input_shape,
                                     include_colorizer_output=include_colorizer_output,
                                     class_to_color=class_to_color)
    else:
        return CombinedGan(generator=generator, critic=critic, input_shape=input_shape,
                           include_colorizer_output=include_colorizer_output)

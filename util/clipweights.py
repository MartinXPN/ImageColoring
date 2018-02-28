from keras.constraints import Constraint
from keras import backend as K


class WeightClip(Constraint):
    """
    Clips the weights incident to each hidden unit to be inside a range [-c; c]
    """
    def __init__(self, clip_min=-2, clip_max=2):
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, p):
        return K.clip(p, self.clip_min, self.clip_max)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'clip_min': self.clip_min,
                'clip_max': self.clip_max}

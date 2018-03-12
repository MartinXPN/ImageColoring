from __future__ import absolute_import
from __future__ import division

from keras import backend as K
from keras.layers import Conv2D


class SubpixelUpSampling(Conv2D):
    def __init__(self, filters, kernel_size, ratio,
                 padding='valid', data_format=None, strides=(1, 1), activation=None,
                 use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 **kwargs):
        super(SubpixelUpSampling, self).__init__(filters=ratio * ratio * filters,
                                                 kernel_size=kernel_size, strides=strides, padding=padding,
                                                 data_format=data_format,
                                                 activation=activation, use_bias=use_bias,
                                                 kernel_initializer=kernel_initializer,
                                                 bias_initializer=bias_initializer,
                                                 kernel_regularizer=kernel_regularizer,
                                                 bias_regularizer=bias_regularizer,
                                                 activity_regularizer=activity_regularizer,
                                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                                 **kwargs)
        self.ratio = ratio

    def _phase_shift(self, inputs):
        r = self.ratio
        bsize, a, b, c = inputs.get_shape().as_list()
        bsize = K.shape(inputs)[0]                                      # Handling (None) type for undefined batch dim
        res = K.reshape(inputs, [bsize, a, b, c // (r * r), r, r])      # bsize, a, b, c/(r*r), r, r
        res = K.permute_dimensions(res, (0, 1, 2, 5, 4, 3))             # bsize, a, b, r, r, c/(r*r)
        # Keras backend does not support tf.split, so in future versions this could be nicer
        res = [res[:, i, :, :, :, :] for i in range(a)]                 # a, [bsize, b, r, r, c/(r*r)
        res = K.concatenate(res, 2)                                     # bsize, b, a*r, r, c/(r*r)
        res = [res[:, i, :, :, :] for i in range(b)]                    # b, [bsize, r, r, c/(r*r)
        res = K.concatenate(res, 2)                                     # bsize, a*r, b*r, c/(r*r)
        return res

    def call(self, inputs, **kwargs):
        return self._phase_shift(super(SubpixelUpSampling, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(SubpixelUpSampling, self).compute_output_shape(input_shape)
        return unshifted[0], self.ratio * unshifted[1], self.ratio * unshifted[2], unshifted[3] // (self.ratio * self.ratio)

    def get_config(self):
        config = super(SubpixelUpSampling, self).get_config()
        if 'rank' in config:            config.pop('rank')
        if 'dilation_rate' in config:   config.pop('dilation_rate')
        config['filters'] //= self.ratio * self.ratio
        config['ratio'] = self.ratio
        return config

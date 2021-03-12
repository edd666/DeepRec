# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-03-11
# @Contact : liaozhi_edo@163.com


"""
    序列处理相关的layer
"""

# packages
import tensorflow as tf
from tensorflow.keras import layers


class SequencePoolingLayer(layers.Layer):
    """
    The SequencePoolingLayer is used to apply pooling operation(sum,mean,max) on variable-length
    sequence feature/multi-value feature.

    Input shape
        - A list of two tensor [seq_value, seq_len]
        - seq_value is a 3D tensor with shape: (batch_size, T, embedding_size)
        - seq_len is a 2D tensor with shape : (batch_size, 1),indicate valid length of each sequence.

    Output shape
        - 3D tensor with shape: (batch_size, 1, embedding_size).
    """

    def __init__(self, mode, mask_zero=False, **kwargs):
        """

        :param mode: str Pooling方法
        :param mask_zero: bool 是否支持mask zero
        :param kwargs:
        :return:
        """
        super(SequencePoolingLayer, self).__init__(**kwargs)
        if mode not in ('sum', 'mean', 'max'):
            raise ValueError("mode must be sum, mean or max")
        self.mode = mode
        self.mask_zero = mask_zero
        self.seq_maxlen = None
        self.eps = tf.constant(1e-8, tf.float32)

    def build(self, input_shape):
        if not self.mask_zero:
            self.seq_maxlen = int(input_shape[0][1])
        super(SequencePoolingLayer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        if self.mask_zero:
            if mask is None:
                raise ValueError("When mask_zero=True,input must support masking.")
            seq_value = inputs
            mask = tf.cast(mask, tf.float32)  # tf.to_float(mask)
            seq_len = tf.math.reduce_sum(mask, axis=-1, keepdims=True)  # (batch_size, 1)
            mask = tf.expand_dims(mask, axis=2)
        else:
            seq_value, seq_len = inputs
            mask = tf.sequence_mask(seq_len, self.seq_maxlen, dtype=tf.float32)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = seq_value.shape[-1]
        mask = tf.tile(mask, [1, 1, embedding_size])

        # max
        if self.mode == 'max':
            seq_value = seq_value - (1 - mask) * 1e9
            return tf.math.reduce_max(seq_value, 1, keepdims=True)

        # sum
        seq_value = tf.math.reduce_sum(seq_value * mask, 1, keepdims=False)

        # mean
        if self.mode == 'mean':
            seq_value = tf.math.divide(seq_value, tf.cast(seq_len, dtype=tf.float32) + self.eps)

        seq_value = tf.expand_dims(seq_value, axis=1)

        return seq_value

    def compute_output_shape(self, input_shape):
        if self.mask_zero:
            embedding_size = input_shape[-1]
        else:
            embedding_size = input_shape[0][-1]

        shape = (None, 1, embedding_size)
        return shape

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {'mode': self.mode, 'mask_zero': self.mask_zero}
        base_config = super(SequencePoolingLayer, self).get_config()
        base_config.update(config)

        return base_config

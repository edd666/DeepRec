# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-03-12
# @Contact : liaozhi_edo@163.com


"""
    基础Layer
"""

# packages
from tensorflow.keras import layers


class DNN(layers.Layer):
    """
    The Multilayer Perceptron.

    Input shape
        - A tensor with shape: (batch_size, ..., input_dim).

    Output shape
        - A tensor with shape: (batch_size, ..., hidden_units[-1]).
    """
    def __init__(self, hidden_units, activation='relu', dropout_rate=0, use_bn=False, **kwargs):
        super(DNN, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        pass

    def input_shape(self):
        pass

    def call(self, inputs, **kwargs):
        pass
    pass
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

    pass
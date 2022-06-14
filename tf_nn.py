# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Date    : 2022/6/10
# @Contact : zhiliao@kugou.net


"""
    TensorFlow常用代码
"""

# packages
import tensorflow as tf
from tensorflow.keras import layers
from collections import namedtuple, OrderedDict
from tensorflow.python.keras.initializers import RandomNormal


class DenseFeat(namedtuple('DenseFeat',
                           ['name', 'dimension', 'dtype'])):
    """
    数值特征
    """
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype='float32'):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash',
                             'dtype', 'embeddings_initializer', 'embedding_name',
                             'trainable'])):
    """
    类别特征
    """
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim, use_hash=False,
                dtype='int32', embeddings_initializer=None, embedding_name=None,
                trainable=True):

        if embedding_name is None:
            embedding_name = name

        if embeddings_initializer is None:
            embeddings_initializer = RandomNormal(mean=0.0, stddev=0.0001, seed=2022)

        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim,
                                              use_hash, dtype, embeddings_initializer,
                                              embedding_name, trainable)

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'length_name',
                                   'combiner', 'weight_name', 'weight_norm'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, length_name, combiner='mean',
                weight_name=None, weight_norm=None):

        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, length_name,
                                                    combiner, weight_name, weight_norm)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embeddings_initializer(self):
        return self.sparsefeat.embeddings_initializer

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def trainable(self):
        return self.sparsefeat.trainable

    def __hash__(self):
        return self.name.__hash__()


def build_input_dict(feature_columns):

    # 1,构建输入字典
    input_dict = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, DenseFeat):
            input_dict[fc.name] = tf.keras.Input(shape=(fc.dimension,), name=fc.name, dtype=fc.dtype)

        elif isinstance(fc, SparseFeat):
            input_dict[fc.name] = tf.keras.Input(shape=(1,), name=fc.name, dtype=fc.dtype)

        elif isinstance(fc, VarLenSparseFeat):
            input_dict[fc.name] = tf.keras.Input(shape=(fc.maxlen,), name=fc.name, dtype=fc.dtype)

            input_dict[fc.length_name] = tf.keras.Input(shape=(1,), name=fc.length_name, dtype='int32')

            if fc.weight_name:
                input_dict[fc.weight_name] = tf.keras.Input(shape=(fc.maxlen, 1), name=fc.weight_name, dtype='float32')
        else:
            raise ValueError('Invalid type in feature columns, got', type(fc))

    return input_dict


def get_dense_value(input_dict, feature_columns):
    """
    获取数值输入

    :param input_dict: dict 输入字典,形如{feature_name: keras.Input()}
    :param feature_columns: list 特征列
    :return:
        dense_value_list: list 数值输入
    """
    # 1,获取DenseFeat
    dense_value_list = list()
    dense_feature_columns = list(filter(
        lambda x: isinstance(x, DenseFeat), feature_columns))
    for fc in dense_feature_columns:
        dense_value_list.append(input_dict[fc.name])

    return dense_value_list


def build_embedding_dict(feature_columns):
    """
    基于特征列(feature columns)构建Embedding字典

    :param feature_columns: list 特征列
    :return:
        embedding_dict: embedding字典,形如{embedding_name: embedding_table}
    """
    # 1,获取SparseFeat和VarLenSparseFeat
    sparse_feature_columns = list(filter(
        lambda x: isinstance(x, SparseFeat), feature_columns))
    varlen_sparse_feature_columns = list(filter(
        lambda x: isinstance(x, VarLenSparseFeat), feature_columns))

    # 2,构建Embedding字典
    embedding_dict = OrderedDict()
    for fc in sparse_feature_columns:
        embedding_dict[fc.embedding_name] = layers.Embedding(input_dim=fc.vocabulary_size,
                                                             output_dim=fc.embedding_dim,
                                                             embeddings_initializer=fc.embeddings_initializer,
                                                             trainable=fc.trainable,
                                                             name='sparse_emb_' + fc.embedding_name)

    for fc in varlen_sparse_feature_columns:
        embedding_dict[fc.embedding_name] = layers.Embedding(input_dim=fc.vocabulary_size,
                                                             output_dim=fc.embedding_dim,
                                                             embeddings_initializer=fc.embeddings_initializer,
                                                             trainable=fc.trainable,
                                                             name='varlen_sparse_emb_' + fc.embedding_name)
    return embedding_dict


def embedding_lookup(input_dict, embedding_dict, query_feature_columns, to_list=False):
    """
    embedding查询

    注意:
        1,query_feature_columns可以是SparseFeat或VarLenSparseFeat
        2,input_dict和embedding_dict必须包含相应的输入和embedding table

    :param input_dict: dict 输入字典,形如{feature_name: keras.Input()}
    :param embedding_dict: embedding字典,形如{embedding_name: embedding_table}
    :param query_feature_columns: list 待查询的特征列
    :param to_list: bool 是否转成list
    :return:
    """
    # 1,查询
    query_embedding_dict = OrderedDict()
    for fc in query_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            raise ValueError('hash embedding lookup has not yet been implemented.')
        else:
            lookup_idx = input_dict[feature_name]
        query_embedding_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)

    if to_list:
        return list(query_embedding_dict.values())

    return query_embedding_dict


class SequencePoolingLayer(layers.Layer):

    def __init__(self, mode, maxlen, **kwargs):
        super(SequencePoolingLayer, self).__init__(**kwargs)
        self.mode = mode
        self.maxlen = maxlen
        pass

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        pass

    pass


def get_varlen_pooling_list(input_dict, embedding_dict, varlen_sparse_feature_columns):
    """
    对序列特征(VarLenSparseFeat)进行Pooling操作
    :param input_dict: dict 输入字典,形如{feature_name: keras.Input()}
    :param embedding_dict: embedding字典,形如{embedding_name: embedding_table}
    :param varlen_sparse_feature_columns: list 序列特征
    :return:
    """
    # 1,对VarLenSparseFeat的embedding进行Pooling操作
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        feature_length_name = fc.length_name
        if fc.weight_name is not None:
            raise ValueError('pooling with weight has not yet been implemented.')
        else:
            pass
    pass

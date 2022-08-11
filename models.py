# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Date    : 2022/6/10
# @Contact : zhiliao@kugou.net


"""
    TensorFlow模型
"""

# packages
from tf_nn import *


def DNN(feature_columns):

    # 1,构建输入字典
    input_dict = build_input_dict(feature_columns)

    # 2,构建embedding dict
    embedding_dict = build_embedding_dict(feature_columns)

    # 3,构建模型输入
    # dense
    dense_value_list = get_dense_value(input_dict, feature_columns)

    # sparse
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
    sparse_embedding_list = embedding_lookup(input_dict, embedding_dict, sparse_feature_columns, to_list=True)

    # seq = varlen sparse
    # pooling
    seq_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns))
    seq_pooling_embedding_list = get_varlen_pooling_list(input_dict, embedding_dict, seq_feature_columns)

    return None

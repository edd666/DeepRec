# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-03-11
# @Contact : liaozhi_edo@163.com


"""
    《Deep Interest Network for Click-Through Rate Prediction》
"""

# packages
from deeprec.layers.utils import concat_func
from deeprec.feature_column import SparseFeat, build_input_dict
from deeprec.inputs import build_embedding_dict, get_dense_value, embedding_lookup, get_varlen_pooling_list


def DIN(feature_columns, query_columns):

    # 1,构建输入字典
    input_dict = build_input_dict(feature_columns)

    # 2,构建Embedding字典
    embedding_dict = build_embedding_dict(feature_columns, seq_mask_zero=True)

    # 3,构建模型的输入
    dense_value_list = get_dense_value(input_dict, feature_columns)

    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
    sparse_embedding_list = embedding_lookup(input_dict, embedding_dict, sparse_feature_columns, to_list=True)

    # varlen sparse = seq
    # pooling
    seq_embedding_list = get_varlen_pooling_list(input_dict, embedding_dict, feature_columns)

    # attention
    query_feature_columns = [fc for fc in feature_columns if fc.name in query_columns]
    query_embedding_list = embedding_lookup(input_dict, embedding_dict, query_feature_columns, to_list=True)
    query = concat_func(query_embedding_list, mask=True)
    keys_columns = ['hist_' + str(col) for col in query_columns]
    keys_feature_columns = [fc for fc in feature_columns if fc.name in keys_columns]
    keys_embedding_list = embedding_lookup(input_dict, embedding_dict, keys_feature_columns, to_list=True)
    keys = concat_func(keys_embedding_list, mask=True)

    pass



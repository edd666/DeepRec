# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-03-11
# @Contact : liaozhi_edo@163.com


"""
    《Deep Interest Network for Click-Through Rate Prediction》
"""

# packages
from deeprec.layers.utils import concat_func
from deeprec.feature_column import SparseFeat, VarLenSparseFeat, build_input_dict
from deeprec.inputs import build_embedding_dict, get_dense_value, embedding_lookup, get_seq_pooling_list


def DIN(feature_columns, behavior_columns):

    # 1,构建输入字典
    input_dict = build_input_dict(feature_columns)

    # 2,构建Embedding字典
    embedding_dict = build_embedding_dict(feature_columns, seq_mask_zero=True)

    # 3,构建模型的输入
    dense_value_list = get_dense_value(input_dict, feature_columns)

    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
    sparse_embedding_list = embedding_lookup(input_dict, embedding_dict, sparse_feature_columns, to_list=True)

    # seq = varlen sparse
    # pooling
    hist_behavior_columns = ['hist_' + str(col) for col in behavior_columns]
    seq_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns))
    seq_pooling_feature_columns = [fc for fc in seq_feature_columns if fc.name not in hist_behavior_columns]
    seq_pooling_list = get_seq_pooling_list(input_dict, embedding_dict, seq_pooling_feature_columns)

    # attention
    
    pass



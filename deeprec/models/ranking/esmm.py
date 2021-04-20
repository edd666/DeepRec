# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-04-20
# @Contact : liaozhi_edo@163.com


"""
    《Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate》
"""

# packages
import tensorflow as tf
from tensorflow.keras import layers
from deeprec.layers.utils import concat_func
from deeprec.layers.sequence import AttentionSequencePoolingLayer
from deeprec.feature_column import SparseFeat, VarLenSparseFeat, build_input_dict
from deeprec.inputs import build_embedding_dict, get_dense_value, embedding_lookup, get_varlen_pooling_list


def ESMM(user_feature_columns, item_feature_columns, behavior_columns,
         att_hidden_units=(64, 16), att_activation='Dice', att_weight_normalization=False):
    """
    ESMM模型

    注意:
        1,feature_columns中特征的相对顺序关系,如item_id,cate_id,其对应的行为序列为
            hist_item_id,hist_item_id.(主要是attention的时候特征要对齐)

    :param user_feature_columns: list 用户特征列
    :param item_feature_columns: list 物品特征列
    :param behavior_columns: list 行为列(attention)
    :param att_hidden_units: tuple Attention中神经元个数
    :param att_activation: str Attention的激活函数
    :param att_weight_normalization: bool Attention中score是否归一化
    :return:
    """
    # 1,构建输入字典
    feature_columns = user_feature_columns + item_feature_columns
    input_dict = build_input_dict(feature_columns)

    # 2,构建Embedding字典
    embedding_dict = build_embedding_dict(feature_columns, seq_mask_zero=True)

    # 3,构建模型的输入
    # Item
    # dense
    item_dense_value_list = get_dense_value(input_dict, item_feature_columns)

    # sparse
    item_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), item_feature_columns))
    item_sparse_feature_columns = [fc for fc in item_sparse_feature_columns if fc.name not in behavior_columns]
    item_sparse_embedding_list = embedding_lookup(input_dict, embedding_dict, item_sparse_feature_columns, to_list=True)

    # concat item input
    item_embedding_input = concat_func(item_sparse_embedding_list, mask=False)
    item_embedding_input = layers.Flatten()(item_embedding_input)
    item_dnn_input = layers.concatenate(item_dense_value_list + [item_embedding_input], axis=-1)

    # User
    # dense
    user_dense_value_list = get_dense_value(input_dict, user_feature_columns)

    # sparse
    user_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), user_feature_columns))
    user_sparse_embedding_list = embedding_lookup(input_dict, embedding_dict,
                                                  user_sparse_feature_columns, to_list=True)

    # seq = varlen sparse
    # pooling
    hist_behavior_columns = ['hist_' + col for col in behavior_columns]
    user_seq_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), user_feature_columns))
    user_seq_pooling_feature_columns = [fc for fc in user_seq_feature_columns if fc.name not in hist_behavior_columns]
    user_seq_pooling_embedding_list = get_varlen_pooling_list(input_dict, embedding_dict,
                                                              user_seq_pooling_feature_columns)

    # attention
    query_feature_columns = [fc for fc in feature_columns if fc.name in behavior_columns]
    query_embedding_list = embedding_lookup(input_dict, embedding_dict, query_feature_columns, to_list=True)
    query = concat_func(query_embedding_list, mask=True)
    hist_feature_columns = [fc for fc in feature_columns if fc.name in hist_behavior_columns]
    keys_embedding_list = embedding_lookup(input_dict, embedding_dict, hist_feature_columns, to_list=True)
    keys = concat_func(keys_embedding_list, mask=True)
    hist = AttentionSequencePoolingLayer(
        hidden_units=att_hidden_units,
        activation=att_activation,
        mask_zero=True,
        weight_normalization=att_weight_normalization,
        return_score=False,
    )([query, keys])

    # concat user input
    user_embedding_input = concat_func(
        user_sparse_embedding_list + user_seq_pooling_embedding_list + [hist],
        mask=False)
    user_embedding_input = layers.Flatten()(user_embedding_input)
    user_dnn_input = layers.concatenate(user_dense_value_list + [user_embedding_input], axis=-1)

    # 4,CTR模型和CVR模型,均为DNN
    ctr_output = ctr_model(user_dnn_input, item_dnn_input)
    cvr_output = cvr_model(user_dnn_input, item_dnn_input)
    ctcvr_output = layers.Multiply(name='ctcvr_output')([ctr_output, cvr_output])

    # 5,functional model
    model = tf.keras.Model(
        inputs=input_dict,
        outputs=[ctr_output, ctcvr_output]
    )

    return model


def ctr_model(user_input, item_input):
    """
    构建CTR模型(DNN)

    :param user_input: tensor 用户输入
    :param item_input: tensor 物品输入
    :return:
    """
    user_feature = layers.Dropout(rate=0.5)(user_input)
    user_feature = layers.BatchNormalization()(user_feature)
    user_feature = layers.Dense(128, activation='relu')(user_feature)
    user_feature = layers.Dense(64, activation='relu')(user_feature)

    item_feature = layers.Dropout(rate=0.5)(item_input)
    item_feature = layers.BatchNormalization()(item_feature)
    item_feature = layers.Dense(128, activation='relu')(item_feature)
    item_feature = layers.Dense(64, activation='relu')(item_feature)

    feature = layers.concatenate([user_feature, item_feature], axis=-1)
    feature = layers.Dropout(rate=0.5)(feature)
    feature = layers.BatchNormalization()(feature)
    feature = layers.Dense(64, activation='relu')(feature)
    ctr_output = layers.Dense(1, activation='sigmoid', name='ctr_output')(feature)

    return ctr_output


def cvr_model(user_input, item_input):
    """
    构建CVR模型(DNN)

    :param user_input: tensor 用户输入
    :param item_input: tensor 物品输入
    :return:
    """
    user_feature = layers.Dropout(rate=0.5)(user_input)
    user_feature = layers.BatchNormalization()(user_feature)
    user_feature = layers.Dense(128, activation='relu')(user_feature)
    user_feature = layers.Dense(64, activation='relu')(user_feature)

    item_feature = layers.Dropout(rate=0.5)(item_input)
    item_feature = layers.BatchNormalization()(item_feature)
    item_feature = layers.Dense(128, activation='relu')(item_feature)
    item_feature = layers.Dense(64, activation='relu')(item_feature)

    feature = layers.concatenate([user_feature, item_feature], axis=-1)
    feature = layers.Dropout(rate=0.5)(feature)
    feature = layers.BatchNormalization()(feature)
    feature = layers.Dense(64, activation='relu')(feature)
    cvr_output = layers.Dense(1, activation='sigmoid', name='cvr_output')(feature)

    return cvr_output

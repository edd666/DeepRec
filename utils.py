# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Date    : 2022/6/10
# @Contact : zhiliao@kugou.net


"""
    Utils
"""

# packages
import time
import datetime
import numpy as np


def get_time_dif(start_time):
    """
    获取时间间隔

    :param start_time: 起始时间
    :return:
    """
    end_time = time.time()
    time_dif = end_time - start_time

    return datetime.timedelta(seconds=int(round(time_dif)))


def build_data_dict(df, varlen_sparse_feature_columns=None):
    """
    DataFrame转换成tf模型输入

    :param df:
    :param varlen_sparse_feature_columns:
    :return:
    """
    # 1,构建模型输入
    data_dict = dict()
    for name, value in df.items():
        value = value.values
        if varlen_sparse_feature_columns and name in varlen_sparse_feature_columns:
            data_dict[name] = np.vstack(value)
        else:
            data_dict[name] = value

    return data_dict

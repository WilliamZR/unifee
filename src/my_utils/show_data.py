# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/10/26 16:27
# Description:
import random
def get_data_examples(data, shuffle = True, max_len = 1000):
    print("function [get_data_examples] may change the instance order in the data list")
    if shuffle:
        random.shuffle(data)
    if max_len is not None:
        return data[:max_len]
    return data

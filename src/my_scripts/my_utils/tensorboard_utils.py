# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/3/23 14:14
# Description:
def add_params_hist(model, tb):
    for i, (name, param) in enumerate(model.named_parameters()):
        tb.add_histgram(name, param)
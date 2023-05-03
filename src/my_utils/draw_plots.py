# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/12/8 14:01
# Description:
import sys
import os
import matplotlib.pyplot as plt

def scalar2number(x, min_scalar = 0, max_scalar = None):
    # if min_scalar is None:
    #     min_scalar = min(x)
    assert min(x) >= 0, min(x)
    if max_scalar is None:
        max_scalar = max(x)
    distri_x = list(range(min_scalar, max_scalar + 1))
    for o in x:
        distri_x[min(max(min_scalar, o), max_scalar)] += 1
    return distri_x


def draw_plot(x, y, xlabel = "", ylabel=""):
    if x is None:
        x = list(range(len(y)))
    plt.plot(x,y)
    # 设置数字标签
    for a, b in zip(x, y):
        plt.text(a, b, round(b,2), ha='center', va='bottom', fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def draw_plot_two(x, y1, y2,  xlabel = "", ylabel=""
                  , legend_labels = None, save_path = None):
    assert len(y1) == len(y2)
    if x is None:
        x = list(range(len(y1)))

    l1, = plt.plot(x,y1, color = "blue")
    l2, = plt.plot(x, y2, color = "#DB7093", linestyle="--")

    if legend_labels:
        #legend_labels = ["SubEvidence Level", "Evidence Level"]
        plt.legend(handles=[l1, l2], labels = legend_labels, loc = "best")

    plt.tick_params(labelsize=15)
    # 设置数字标签
    for a, b in zip(x, y1):
        plt.text(a, b, round(b,2), ha='center', va='bottom', fontsize=12)
    for a, b in zip(x, y2):
        plt.text(a, b, round(b,2), ha='center', va='bottom', fontsize=12)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def draw_bar_two(x, y1, y2,  xlabel = "", ylabel="", xticks = None, legend_labels = None, save_path = None, ylim = None):
    assert len(y1) == len(y2)
    if x is None:
        x = list(range(len(y1)))

    l1 = plt.bar(x,y1, color = "skyblue")
    l2 = plt.bar(x, y2, color = "#DB7093")
    if legend_labels:
        # legend_labels = ["EDGN", "w/o Decomposition"]
        plt.legend(handles=[l1, l2], labels = legend_labels, loc = "best", fontsize=13)
    # 设置数字标签
    for a, b in zip(x, y1):
        plt.text(a, b, round(b,2), ha='center', va='bottom', fontsize=11)
    for a, b in zip(x, y2):
        plt.text(a, b, round(b,2), ha='center', va='bottom', fontsize=11)

    if ylim:
        # ylim = (85, 95)
        plt.ylim(ylim)
    # xticks = ["0~20", "20~40", "40~60", "60~80", ">80"]
    plt.xticks(x, xticks, fontsize=11)

    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    if save_path:
        plt.savefig(save_path)
    plt.show()
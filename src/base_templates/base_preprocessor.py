# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/12/13 20:17
# Description:

import os

class BasePreprocessor(object):
    def __init__(self, args):
        self.args = args
        self.train_data = []
        self.valid_data = []
        self.test_data = []

        self.data_generator = None

    def process(self, input_dir, output_dir, data_generator, tokenizer, dataset = ["train", "dev", "test"]):
        args = self.args
        self.data_generator = data_generator

        if "dev" in dataset:
            self.valid_data = data_generator(
                os.path.join(input_dir, "dev.jsonl"), tokenizer, os.path.join(output_dir, "dev"), "dev", args)

        if "dev_retrieved" in dataset:
            self.valid_data = data_generator(
                # os.path.join(input_dir, "dev.combined.not_precomputed.p5.s5.t3.cells.jsonl")
                os.path.join(input_dir, "dev.fusion.results.jsonl")
                , tokenizer, os.path.join(output_dir, "dev"), "dev_retrieved", args)

        if "train" in dataset:
            self.train_data = data_generator(
                os.path.join(input_dir, "train.jsonl"), tokenizer, os.path.join(output_dir, "train"), "train", args)

        if "test" in dataset:
            self.test_data = data_generator(
                os.path.join(input_dir, "test.jsonl")
                , tokenizer, os.path.join(output_dir, "test"), "test", args)

        print("train data length:", len(self.train_data))
        print("dev data length:", len(self.valid_data))
        print("test data length:", len(self.test_data))

        return self.train_data, self.valid_data, self.test_data
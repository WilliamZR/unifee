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
                os.path.join(input_dir, "dev.pos_neg.sentences.jsonl"), tokenizer, output_dir, "dev", args)

        if "train" in dataset:
            self.train_data = data_generator(
                os.path.join(input_dir, "train.pos_neg.sentences.jsonl"), tokenizer, output_dir, "train", args)

        if "test" in dataset:
            self.test_data = data_generator(
                os.path.join(input_dir, "test.pos_neg.sentences.jsonl"), tokenizer, output_dir, "test", args)

        print("train data length:", len(self.train_data))
        print("dev data length:", len(self.valid_data))
        print("test data length:", len(self.test_data))

        return self.train_data, self.valid_data, self.test_data

import random
from torch.utils.data.dataset import Dataset
import torch
from my_utils import load_jsonl_data, refine_obj_data
from tqdm import tqdm


class RobertaSentenceGenerator(Dataset):
    def __init__(self, input_path, tokenizer, cache_dir, data_type, args):
        super(RobertaSentenceGenerator, self).__init__()
        self.model_name = str(type(self))
        self.args = args
        self.config = args.config
        self.data_type = data_type
        self.raw_data = self.preprocess_raw_data(self.get_raw_data(input_path, keys=self.get_refine_keys()))
        if args.test_mode or (not "train" in self.data_type):
            self.cand_id_lst = self.get_cand_ids()

        self.tokenizer = args.tokenizer
        self.max_seq_len = 512
        self.generate_train_instances_one_epoch()

    def generate_train_instances_one_epoch(self):
        self.instances = []
        for entry in self.raw_data:
            if self.args.test_mode or (not "train" in self.data_type):
                for cand in entry["all_candidates"]:
                    self.instances.append([entry["claim"], cand[1] + " : " + cand[0], None])
            else:
                pos_sents = entry["all_pos_sents"]
                neg_sents = entry["all_neg_sents"]
                if (not pos_sents) or (not neg_sents):
                    continue
                for ps in pos_sents:
                    # [sent, page_title, sent_id]
                    ns = random.choice(neg_sents)
                    self.instances.append([entry["claim"], ps[1] + " : " + ps[0], ns[1] + " : " + ns[0]])
        print("generate {} sentence pairs".format(len(self.instances)))

        if self.args.test_mode or (not "train" in self.data_type):
            assert len(self.instances) == sum([len(ci) for ci in self.cand_id_lst]) \
                , print(len(self.instances), sum([len(ci) for ci in self.cand_id_lst]))

    def get_cand_ids(self):
        cand_id_lst = []
        for entry in self.raw_data:
            cand_id_lst.append([cd[1] + "_" + cd[2] for cd in entry["all_candidates"]])
        return cand_id_lst

    def print_example(self):
        pass
        # instance = self.raw_data[0]
        # for k, v in instance.items():
        #     print(k, " : ", v)
        #
        # instance = self.raw_data[-1]
        # for k, v in instance.items():
        #     print(k, " : ", v)

    def preprocess_raw_data(self, raw_data):
        return raw_data

    def get_refine_keys(self):
        keys = None
        return keys

    def get_raw_data(self, input_path, keys=None):
        raw_data = load_jsonl_data(input_path)
        if keys is not None:
            raw_data = refine_obj_data(raw_data, keys)
        return raw_data

    def get_encodings(self, s1, s2):
        pad_idx = self.tokenizer.pad_token_id
        max_len = self.max_seq_len

        if s2 is None:
            input_ids = [2,2]
            input_mask = [1,1]
        else:
            encodes = self.tokenizer(s1, s2)
            input_ids = encodes["input_ids"][:self.max_seq_len]
            input_mask = [1] * len(input_ids)
            input_ids += [pad_idx] * (max_len - len(input_ids))
            input_mask += [0] * (max_len - len(input_mask))
        return input_ids, input_mask

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        # raw_data = self.raw_data[idx]
        instance = self.instances[idx]
        pos_ids, pos_attention_mask = self.get_encodings(instance[0], instance[1])
        pos_ids = torch.tensor(pos_ids).to(self.args.device)
        pos_attention_mask = torch.tensor(pos_attention_mask).to(self.args.device)

        neg_ids, neg_attention_mask = self.get_encodings(instance[0], instance[2])
        neg_ids = torch.tensor(neg_ids).to(self.args.device)
        neg_attention_mask = torch.tensor(neg_attention_mask).to(self.args.device)

        return pos_ids, pos_attention_mask, neg_ids, neg_attention_mask

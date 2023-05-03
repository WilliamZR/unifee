# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/12/14 22:27
# Description:

from base_templates import BaseGenerator
import torch
import numpy as np

def collate_fn(batch):
    raw_data, input_ids, input_mask, label = map(list, zip(*batch))

    batched_raw_data = raw_data
    batch_input_ids = torch.stack(input_ids)
    batch_input_mask = torch.stack(input_mask)
    batched_label = torch.stack(label)

    return batched_raw_data, batch_input_ids, batch_input_mask, batched_label

class RobertaGenerator(BaseGenerator):
    def __init__(self, input_path, tokenizer, cache_dir, data_type, args):
        super(RobertaGenerator, self).__init__(input_path, data_type, args)
        assert len(self.labels) != []
        assert 'roberta' in args.bert_name
        self.model_name = str(type(self))

        self.tokenizer = tokenizer
        self.input_ids_lst = []
        self.get_flat_tokens(self.raw_data, cache_dir)
        assert len(self.labels) == len(self.input_ids_lst)

        self.input_mask = []
        self.input_ids = []
        self.get_seq_inputs_from_inputs_lst(tokenizer, max_len=self.args.max_seq_len)
        assert len(self.labels) == len(self.input_ids)
        assert len(self.labels) == len(self.input_mask)

    def get_flat_tokens(self, data, cache_dir):
        assert False, "RobertaGenerator::get_flat_tokens ,you shouldn't go here!"

    def get_plm_inputs_from_input_ids(self, input_ids, tokenizer, max_len=512):
        if isinstance(input_ids[0], list):
            flat_input_ids = []
            for ids in input_ids:
                flat_input_ids.extend(ids)
            input_ids = flat_input_ids

        pad_idx = tokenizer.pad_token_id
        input_mask = [1] * len(input_ids)
        input_ids += [pad_idx] * (max_len - len(input_ids))
        input_mask += [0] * (max_len - len(input_mask))
        return input_ids, input_mask

    def get_seq_inputs_from_inputs_lst(self, tokenizer, max_len):
        for idl in zip(self.input_ids_lst):
            input_ids, input_mask = self.get_plm_inputs_from_input_ids(idl, tokenizer, max_len)
            self.input_ids.append(input_ids)
            self.input_mask.append(input_mask)

    def get_plm_inputs_lst(self, sent_a, sent_b_lst, tokenizer, max_len=512, return_mask=None):
        """
        :param sent_b_lst:
        :param sent_a:
        :param tokenizer:
        :param max_len:
        :param return_mask: ["evi_mask_cls", "mask_a", "mask_b", "word_mask_a", "word_mask_b"]
        :return:
        """

        allowed_return_mask = ["word_mask_a", "word_mask_b"]

        input_ids_lst = []
        input_ids, tokens_a = self.encode_one_sent(sent_a, tokenizer, is_first_sent=True)

        input_ids_lst.append(input_ids)
        len_a = len(input_ids)
        len_b_lst = []
        tokens_b_lst = []

        if sent_b_lst is not None:
            for sent_b in sent_b_lst:
                input_ids, tokens_b = self.encode_one_sent(sent_b, tokenizer, is_first_sent=False)
                len_b = len(input_ids)
                len_b_lst.append(len_b)
                input_ids_lst.append(input_ids)
                tokens_b_lst.append(tokens_b)

        assert len(input_ids) <= max_len
        assert sum(len_b_lst) + len_a == sum([len(ids) for ids in input_ids_lst])

        mask_dict = {"seq_len_a": len_a, "seq_len_b": len_b_lst}

        if return_mask is not None:
            for rm in return_mask:
                if rm not in allowed_return_mask:
                    assert False, print(rm)

            if "mask_a" in return_mask:
                mask_dict["mask_a"] = self.get_bpe_mask_roberta(0, len_a, max_len)
            if "mask_b" in return_mask:
                mask_dict["mask_b"] = []
                stat = len_a
                for len_b in len_b_lst:
                    mask_dict["mask_b"].append(self.get_bpe_mask_roberta(stat, len_b, max_len))
                    stat += len_b
            if "evi_mask_cls" in return_mask:
                mask_cls = []
                cls_idx = len_a + 1
                for lb in len_b_lst:
                    mask_cls.append(cls_idx)
                    cls_idx += lb
                mask_dict["evi_mask_cls"] = mask_cls

            word_num_a = self.get_word_num_from_roberta_tokens(tokens_a)
            word_num_b_lst = [self.get_word_num_from_roberta_tokens(tokens_b) for tokens_b in tokens_b_lst]

            if "word_mask_a" in return_mask:
                total_word_num = word_num_a
                if "word_mask_b" in return_mask:
                    total_word_num += sum(word_num_b_lst)
                mask_mat = self.get_word_mask_roberta(tokens_a, 0, word_num_a, len_a, total_word_num)
                mask_dict["word_mask_a"] = mask_mat
                mask_dict["word_num_a"] = word_num_a

            if "word_mask_b" in return_mask:
                total_word_num = word_num_a + sum(word_num_b_lst)
                mask_dict["word_mask_b"] = []
                mask_dict["word_num_b"] = []
                stat = word_num_a
                for tokens_b, len_b, word_num_b in zip(tokens_b_lst, len_b_lst, word_num_b_lst):
                    mask_mat = self.get_word_mask_roberta(tokens_b, stat, word_num_b, len_b, total_word_num)
                    mask_dict["word_mask_b"].append(mask_mat)
                    mask_dict["word_num_b"].append(word_num_b)
                    stat += word_num_b
        return input_ids_lst, mask_dict

    def encode_one_sent(self, sent, tokenizer, is_first_sent=False):
        if isinstance(sent, str):
            tokens = tokenizer.tokenize(sent)
        else:
            tokens = sent

        if is_first_sent:
            tokens_encode = ["<s>"]
        else:
            tokens_encode = ["</s>"]

        tokens_encode.extend(tokens)
        tokens_encode.append("</s>")
        input_ids = tokenizer.convert_tokens_to_ids(tokens_encode)
        return input_ids, tokens

    def get_bpe_mask_roberta(self, stat, seq_len, max_len):
        mask_mat = np.zeros([seq_len - 2, max_len], dtype=np.float32)
        mask_mat[:, stat + 1:stat + seq_len - 1] = np.eye(seq_len - 2, dtype=np.float32)
        return mask_mat

    def get_cls_mask_roberta(self, seq_len_lst, max_len):
        sent_num = len(seq_len_lst)
        assert sent_num >= 1
        mask_mat = np.zeros([sent_num, max_len], dtype=np.float32)
        stat = 0
        for idx, sl in enumerate(seq_len_lst):
            if sl is None:
                break
            mask_mat[idx][stat + 1] = 1
            stat += sl
        return mask_mat

    def get_word_num_from_roberta_tokens(self, tokens):
        if tokens:
            word_num = len([token for token in tokens if token.startswith("Ġ")]) + 1
        else:
            word_num = 0
        return word_num

    def get_word_mask_roberta(self, tokens, stat, word_num, seq_len, total_word_num):
        word_len_lst = []
        word_len = 0
        assert seq_len == len(tokens) + 2
        for token in tokens:
            if token.startswith("Ġ"):
                word_len_lst.append(word_len)
                word_len = 1
            else:
                word_len += 1
        if word_len != 0:
            word_len_lst.append(word_len)
        assert len(word_len_lst) == word_num

        mask_mat = np.zeros([total_word_num, len(tokens)+2], dtype=np.float32)
        ptr = 1
        for idx, word_len in enumerate(word_len_lst):
            for _ in range(word_len):
                mask_mat[stat + idx][ptr] = 1.0 / word_len
                ptr += 1
        assert ptr == len(tokens)+1

        return mask_mat

    def __getitem__(self, idx):
        raw_data = self.raw_data[idx]
        input_ids = torch.tensor(self.input_ids[idx]).to(self.args.device)
        input_mask = torch.tensor(self.input_mask[idx]).to(self.args.device)
        label = torch.tensor(self.labels[idx]).to(self.args.device)

        return raw_data, input_ids, input_mask, label



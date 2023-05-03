# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/8/26 20:14
# Description:
from .common_utils import load_json_data, load_jsonl_data
import config
import numpy as np

def compare_results(path1, ckpt1, path2, ckpt2):
    '''
    :param path1: data path of the first model (true)
    :param ckpt1: checkpoint name of the first model
    :param path2: data path of the second model (false)
    :param ckpt2: checkpoint name of the second model
    :return: a tuple, (data1_slct, data2_slct)
    '''
    data1 = load_jsonl_data(path1)
    data2 = load_jsonl_data(path2)
    preds1 = load_json_data(f"./checkpoints/{ckpt1}/ckpt.meta")["val_preds"]
    preds2 = load_json_data(f"./checkpoints/{ckpt2}/ckpt.meta")["val_preds"]
    labels = [config.label2idx[entry["label"]] for entry in data1]
    idx_slct = [idx for idx, (p1, p2, lb) in enumerate(zip(preds1, preds2, labels)) if p1 == lb and p2 != lb]
    data1_slct = []
    data2_slct = []
    for idx in idx_slct:
        data1[idx]["preds"] = config.idx2label[preds1[idx]]
        data1_slct.append(data1[idx])
        data2[idx]["preds"] = config.idx2label[preds2[idx]]
        data2_slct.append(data2[idx])
    return data1_slct, data2_slct

def false_results(data, ckpt):
    '''
    :param data: instance list or the data path of a model
    :param ckpt: checkpoint name
    :return: list of false instances
    '''
    if isinstance(data, str):
        data = load_jsonl_data(data)
    preds = load_json_data(f"./checkpoints/{ckpt}/ckpt.meta")["val_preds"]
    labels = [config.label2idx[entry["label"]] for entry in data]
    odata = []
    for entry, p, l in zip(data, preds, labels):
        if p != l:
            entry["preds"] = config.idx2label[p]
            odata.append(entry)
    return odata

def get_evi_len_curve(data, ckpt):
    def get_tf_labels(data, ckpt):
        if isinstance(data, str):
            data = load_jsonl_data(data)
        preds = load_json_data(f"./checkpoints/{ckpt}/ckpt.meta")["val_preds"]
        labels = [config.label2idx[entry["label"]] for entry in data]
        tf_labels = [1 if p == l else 0 for p,l in zip(preds, labels)]
        return tf_labels

    def get_evi_length(data):
        if isinstance(data, str):
            data = load_jsonl_data(data)
        evi_len_lst = []
        for entry in data:
            evi_lens = [len(f"{e_t} : {e}".split()) for e,e_t in zip(entry["full_evidences"], entry["full_evidences_title"])]
            if evi_lens:
                evi_len_lst.append(sum(evi_lens)*1.0/len(evi_lens))
            else:
                evi_len_lst.append(0)
        return evi_len_lst

    def get_golden_evi_length(data):
        if isinstance(data, str):
            data = load_jsonl_data(data)
        evi_len_lst = []
        for entry in data:
            evi_lens = [len(f"{e_t} : {e}".split())
                        for e,e_t, e_l in zip(entry["full_evidences"], entry["full_evidences_title"], entry["full_evidences_label"]) if e_l == 1]
            if evi_lens:
                evi_len_lst.append(sum(evi_lens)*1.0/len(evi_lens))
            else:
                evi_len_lst.append(0)
        return evi_len_lst

    if isinstance(data, str):
        data = load_jsonl_data(data)
    tf_labels = get_tf_labels(data, ckpt)
    evi_len_lst = get_golden_evi_length(data)
    assert len(tf_labels) == len(evi_len_lst)
    print(len(tf_labels))
    true_evi_lens = [e_l for e_l, tfl in zip(evi_len_lst, tf_labels) if tfl == 1]
    false_evi_lens = [e_l for e_l, tfl in zip(evi_len_lst, tf_labels) if tfl == 0]

    true_evi_lens = [el for el in true_evi_lens if el != 0]
    false_evi_lens = [el for el in false_evi_lens if el != 0]

    print("true evi lens avg:", sum(true_evi_lens)/len(true_evi_lens))
    print("false evi lens avg:", sum(false_evi_lens)/len(false_evi_lens))


def result_distri(ckpt, data_dir = "new_data"):
    def cal_class(labels):
        nums = [0, 0, 0]
        for label in labels:
            nums[label] += 1
        for idx, num in enumerate(nums):
            print(config.idx2label[idx], num)
        print('\n')
        return nums

    def TFN(preds, golds):
        tb = [[0, 0, 0] for _ in range(3)]
        for p, g in zip(preds, golds):
            tb[p][g] += 1
        for line in tb:
            print(line)
        return tb

    def print_P_R(tb):
        tb = np.array(tb)
        labels = ["SUPPORTS", "REFUTES ", "NEI     "]
        print("\n         Pr     Rc    F1    ")
        for i in range(len(tb)):
            print(labels[i], end=' ')
            pr = round(tb[i][i] * 1.0 / sum(tb[i, :]), 4)
            rc = round(tb[i][i] * 1.0 / sum(tb[:, i]), 4)
            f1 = round((pr + rc) / 2, 4)
            print(pr, rc, f1)

    input_path = f"./data/{data_dir}/valid.jsonl"
    data = load_jsonl_data(input_path)
    preds = load_json_data(f"./checkpoints/{ckpt}/ckpt.meta")["val_preds"]
    golds = [config.label2idx[entry['label']] for entry in data]
    acc = sum([1 for p, g in zip(preds, golds) if p == g]) * 1.0 / len(preds)
    cal_class(golds)
    cal_class(preds)
    print("acc:", acc)
    tb = TFN(preds, golds)
    print_P_R(tb)

def get_golden_evidences():
    data = load_jsonl_data("./data/new_data/valid_temp.jsonl")
    ori_data = load_jsonl_data("./data/new_data/valid.jsonl")
    gold_evis_data = []
    for entry, ori_entry in zip(data, ori_data):
        gold_evidences = entry["evidence"]
        oentry = {"id": entry["id"], "golden_evidences": []}
        gold_sets = []
        for gold_evidence in gold_evidences:
            gold_set = set()
            for g_evi in gold_evidence:
                gold_set.add((g_evi[2], g_evi[3]))
            gold_sets.append(gold_set)
        pred_evidences = [[(evi[0], evi[1]), evi_idx, evi] for evi_idx, evi in enumerate(ori_entry["evidence"])]
        for pe in pred_evidences:
            for gold_set in gold_sets:
                if pe[0] in gold_set:
                    oentry["golden_evidences"].append([pe[2][0].relpace("_", " "), pe[1], pe[2][2]])
                    break
        gold_evis_data.append(oentry)
    return gold_evis_data

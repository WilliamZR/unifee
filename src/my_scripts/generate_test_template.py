# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2022/1/4 19:46
# Description:
import os
from tqdm import tqdm
os.chdir("/home/hunan/feverous/mycode")

from my_utils import load_jsonl_data, save_jsonl_data

# input_path = "data/dev.combined.not_precomputed.p5.s5.t3.cells.jsonl"
# output_path = "data/dev_temp.jsonl"
# data = load_jsonl_data(input_path)[1:]
def generate_dev_submission():
    input_path = "data/dev.combined.not_precomputed.p5.s5.t3.cells.verdict.jsonl"
    output_path = "submissions/submission_fusion_baseline_BertCls_0418_18:59:23.dev.jsonl"
    data = load_jsonl_data(input_path)

    odata = []
    for entry in data:
        oentry = {}
        oentry["predicted_label"] = entry.get("predicted_label", "SUPPORTS")
        predicted_evidence = list(set(entry["predicted_evidence"]))
        new_predicted_evidence = [[el.split('_')[0], el.split('_')[1] if 'table_caption' not in el and 'header_cell' not in el else '_'.join(el.split('_')[1:3]), '_'.join(el.split('_')[2:]) if 'table_caption' not in el and 'header_cell' not in el else '_'.join(el.split('_')[3:])] for el in predicted_evidence]
        oentry["predicted_evidence"] = new_predicted_evidence
        odata.append(oentry)
    save_jsonl_data(odata, output_path)

def generate_test_template():
    # input_path = "data/test.combined.not_precomputed.p5.s5.t3.cells.jsonl"
    input_path = "data/test.extracted.jsonl"
    output_path = "data/test_temp.jsonl"
    data = load_jsonl_data(input_path)

    odata = []
    for entry in data[1:]:
        oentry = {}
        oentry["predicted_label"] = entry.get("predicted_label", "SUPPORTS")
        predicted_evidence = list(set(entry["predicted_evidence"]))
        new_predicted_evidence = [[el.split('_')[0], el.split('_')[
            1] if 'table_caption' not in el and 'header_cell' not in el else '_'.join(el.split('_')[1:3]), '_'.join(
            el.split('_')[2:]) if 'table_caption' not in el and 'header_cell' not in el else '_'.join(
            el.split('_')[3:])] for el in predicted_evidence]
        oentry["predicted_evidence"] = new_predicted_evidence
        odata.append(oentry)
    save_jsonl_data(odata, output_path)


def merge_test_results():
    input_path = "data/test_temp.jsonl"
    data = load_jsonl_data(input_path)
    label_path = "submissions/submission_fusion_baseline_BertCls_0415_21:15:46.jsonl"
    label_data = load_jsonl_data(label_path)
    test_preds = [entry["predicted_label"] for entry in label_data]
    assert len(test_preds) == len(data)
    for entry, pred in zip(data, test_preds):
        entry["predicted_label"] = pred
    save_jsonl_data(data, label_path)

# generate_test_template()
# merge_test_results()
if __name__ == '__main__':
    generate_dev_submission()
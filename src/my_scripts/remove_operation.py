# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/12/23 14:54
# Description:

import os
from tqdm import tqdm
from my_utils import load_jsonl_data, save_jsonl_data
os.chdir("../../")

for s in ["dev"]: #, "train"]:
    # os.system(f"mv data/{s}.combined.not_precomputed.p5.s5.t3.jsonl data/{s}.combined.not_precomputed.p5.s5.t3.jsonl.bk")
    input_path = f"data/{s}.combined.not_precomputed.p5.s5.t3.jsonl.bk"
    output_path = f"data/{s}.combined.not_precomputed.p5.s5.t3.jsonl"
    data = load_jsonl_data(input_path)
    for entry in tqdm(data[1:]):
        entry.pop('annotator_operations')
    save_jsonl_data(data, output_path)

from baseline.retriever.eval_sentence_retriever import eval_sentence_obj
from my_utils import load_jsonl_data, save_jsonl_data, load_jsonl_one_line
import os
os.chdir("/home/hunan/feverous/mycode")
def main():
    for data_type in ["dev", "train", "test"]:
    # for data_type in ["dev"]:
        input_path = f"/home/hunan/feverous/mycode/data/{data_type}.sentences.roberta.p5.s5.jsonl.RobertaCls_0412_16:25:22"
        output_path = f"/home/hunan/feverous/mycode/data/{data_type}.sentences.roberta.p5.s5.jsonl"
        data = load_jsonl_data(input_path)
        if data_type != "test":
            eval_sentence_obj(data, data_type)
        odata = [{"id":''}]
        for entry in data:
            oentry = {"id":entry["id"], "claim": entry["claim"]}
            sent_scores = entry["sentence_scores"]
            sent_scores.sort(key=lambda x: float(x[1]), reverse=True)
            oentry["predicted_sentences"] = [s[0] for s in sent_scores[:5]]
            odata.append(oentry)
        if data_type != "test":
            eval_sentence_obj(odata, data_type, sent_num=5)
        save_jsonl_data(odata, output_path)


if __name__ == "__main__":
    # data_type = "dev"
    # input_path = f"/home/hunan/feverous/mycode/{data_type}.sentences.roberta.p5.s5.jsonl.RobertaCls_0412_16:25:22"
    # entry = load_jsonl_one_line(input_path)
    main()

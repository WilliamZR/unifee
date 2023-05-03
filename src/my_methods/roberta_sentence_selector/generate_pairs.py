import argparse
import json
import os
from my_utils import load_jsonl_data
from utils.log_helper import LogHelper
from tqdm import tqdm
from baseline.drqa.retriever.doc_db import DocDB
from utils.wiki_page import WikiPage
from utils.util import JSONLineReader
import unicodedata



def get_all_sentences(page):
    page = unicodedata.normalize('NFD', page)
    # lines = db.get_doc_lines(page)
    try:
        lines = json.loads(db.get_doc_json(page))
    except:
        print(f"{page} page not loaded")
        return []
    current_page = WikiPage(page, lines)
    all_sentences = current_page.get_sentences()
    sentences = [str(sent) for sent in all_sentences]
    sentence_ids = [sent.get_id() for sent in all_sentences]

    return list(zip(sentences, [page] * len(lines), sentence_ids))

def generate_lines(line, gold_line, split):
    #print(f'Line:{line}')

    id = gold_line["id"]
    claim = gold_line["claim"]

    instances = []
    gold_evi_by_page = {}

    if "evidence" in gold_line:
        gold_evidences = [evi for evi_set in gold_line["evidence"] for evi in evi_set["content"] if "_sentence_" in evi]
        gold_evidences = list(set(gold_evidences))
        for evi in gold_evidences:
            title = evi.split("_")[0]
            sent_id = '_'.join(evi.split("_")[1:])
            if title in gold_evi_by_page:
                gold_evi_by_page[title].append(sent_id)
            else:
                gold_evi_by_page[title] = [sent_id]

        for page in gold_evi_by_page:
            sentences = get_all_sentences(page)
            if not sentences:
                continue
            pos_sents = [s for s in sentences if s[2] in gold_evi_by_page[page]]
            neg_sents = [s for s in sentences if not s[2] in gold_evi_by_page[page]]

            instances.append(
                {
                    "page": page,
                    "pos_sents": pos_sents,
                    "neg_sents": neg_sents
                }
            )

    retrieved_instances = []
    if 'predicted_pages' in line:
        sorted_p = list(sorted(line['predicted_pages'], reverse=True, key=lambda elem: elem[1]))
        pages = [p[0] for p in sorted_p[:args.max_page]]
        pages = [p for p in pages if p not in gold_evi_by_page]

        for page in pages:
            neg_sents = get_all_sentences(page)
            retrieved_instances.append(
                {
                    "page": page,
                    "neg_sents": neg_sents
                }
            )

    all_pos_sents = []
    all_neg_sents = []
    for page in instances:
        all_pos_sents.extend(page["pos_sents"])
        all_neg_sents.extend(page["neg_sents"])

    for page in retrieved_instances:
        all_neg_sents.extend(page["neg_sents"])

    oentry = {
        "id": id,
        "claim": claim,
        "instances": instances,
        "retrieved_instances": retrieved_instances,
        "all_pos_sents": all_pos_sents,
        "all_neg_sents": all_neg_sents,
    }

    # if split != "train":
    all_candidates = []
    sorted_p = list(sorted(line['predicted_pages'], reverse=True, key=lambda elem: elem[1]))
    pages = [p[0] for p in sorted_p[:args.max_page]]

    for page in pages:
        candidate_sents = get_all_sentences(page)
        all_candidates.extend(candidate_sents)
    oentry["all_candidates"] = all_candidates

    return oentry

if __name__ == "__main__":
    LogHelper.setup()
    LogHelper.get_logger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument('--db', type=str, default="data/feverous_wikiv1.db", help='/path/to/saved/db.db')
    parser.add_argument('--max_page',type=int, default=5)
    parser.add_argument('--max_sent',type=int, default=5)
    parser.add_argument('--data_path',type=str, default="data")
    # parser.add_argument('--split', type=str)

    args = parser.parse_args()

    db = DocDB(args.db)

    # print(db.get_doc_ids())

    # for split in ["debug", "dev", "train", "test"]:
    for split in ["train"]:
        args.split = split
        jlr = JSONLineReader()

        gold_input_path = "{0}/{1}.jsonl".format(args.data_path, args.split)
        gold_data = load_jsonl_data(gold_input_path)[1:]

        input_path = "{0}/{1}.pages.p{2}.jsonl".format(args.data_path, args.split, args.max_page)
        output_path = "{0}/{1}.pos_neg.sentences.jsonl".format(args.data_path, args.split)

        with open(input_path,"r") as f, open(output_path, "w") as out_file:
            lines = jlr.process(f)
            assert len(lines) == len(gold_data), print(len(lines), len(gold_data))
            for line, gold_line in tqdm(zip(lines, gold_data), total=len(gold_data), desc="processing {} split".format(args.split)):
                #print(f'line:{line}')
                line = generate_lines(line, gold_line, split)
                #print(f'tfidf_line:{line}')
                out_file.write(json.dumps(line) + "\n")


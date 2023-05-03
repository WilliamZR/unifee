# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/12/17 13:36
# Description:

import argparse
import os
import string

from my_utils import load_jsonl_data
from baseline.drqa.retriever import DocDB
from tqdm import tqdm
from cleantext import clean
from urllib.parse import unquote
import unicodedata
import re

from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh import qparser

# pt = re.compile(r"\[\[.*?\|(.*?)]]")

def clean_text(text):
    # text = re.sub(pt, r"\1", text)
    text = unquote(text)
    text = unicodedata.normalize('NFD', text)
    text = clean(text.strip(),fix_unicode=True,               # fix various unicode errors
    to_ascii=False,                  # transliterate to closest ASCII representation
    lower=False,                     # lowercase text
    no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                  # replace all URLs with a special token
    no_emails=False,                # replace all email addresses with a special token
    no_phone_numbers=False,         # replace all phone numbers with a special token
    no_numbers=False,               # replace all numbers with a special token
    no_digits=False,                # replace all digits with a special token
    no_currency_symbols=False,      # replace all currency symbols with a special token
    no_punct=False,                 # remove punctuations
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number="<NUMBER>",
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    lang="en"                       # set to 'de' for German special handling
    )
    return text

# import nltk
# ignored_words = set(nltk.corpus.stopwords.words('english'))
# punct_set = set(['.', "''", '``', ',', '(', ')'] + list(string.punctuation))
# ignored_words = ignored_words.union(punct_set)
#
# import nltk.stem
# stemmizer = nltk.stem.SnowballStemmer('english')
#
# def tokenize(text):
#     tokens = nltk.word_tokenize(text)
#     tokens = [stemmizer.stem(tk) for tk in tokens if tk.lower() not in ignored_words]
#     return tokens


def main(args, doc_db):
    cand_ids = doc_db.get_doc_ids()
    print(len(cand_ids))
    # cand_ids = [cand_id for cand_id in cand_ids if not cand_id.isdigit()]
    # cand_ids = [cand_id for cand_id in cand_ids if len(cand_id) > 3]
    # print(len(cand_ids))

    index_dir = "whoosh_indexdir"
    if os.path.exists(index_dir):
        ix = open_dir(index_dir)
    else:
        os.mkdir(index_dir)
        schema = Schema(title=TEXT(stored=True), content=TEXT)
        ix = create_in(index_dir, schema)
        writer = ix.writer()
        for idx, cand_id in enumerate(cand_ids): #tqdm(enumerate(cand_ids)):
            title = cand_id
            doc = ''.join([cand_id + " " for _ in range(3)]) + clean_text(doc_db.get_doc_text(cand_id))
            # doc = ' '.join(tokenize(doc)[:128])
            writer.add_document(title=title, content=doc)
            if idx % 10000 == 0:
                print(idx)
                # writer.commit()
        writer.commit()

    args.input_path = "data/{}.pages.p{}.jsonl".format(args.split, args.count)
    data = load_jsonl_data(args.input_path)

    predicted_pages = []

    with ix.searcher() as searcher:
        for idx, js in tqdm(enumerate(data)):
            query = QueryParser('content', ix.schema, group=qparser.OrGroup).parse(js["claim"])
            res_pages = searcher.search(query)[:10]
            res_pages = [o["title"] for o in res_pages]
            oentry = {
                "id": js["id"],
                "predicted_pages": list(zip(res_pages, list(range(len(res_pages)))))
            }
            predicted_pages.append(oentry)
            # if idx and (idx%1000) == 0:
            #     page_coverage_obj(args, predicted_pages, max_predicted_pages=5)

    # from my_utils import save_jsonl_data
    # save_jsonl_data(predicted_pages, f"./{args.split}.pages.whoosh_bm25.p10.jsonl")

    from baseline.retriever.eval_doc_retriever import page_coverage_obj
    page_coverage_obj(args, predicted_pages, max_predicted_pages=5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default="dev")
    parser.add_argument('--count', type=int, default=150)
    args = parser.parse_args()
    # os.chdir("{dir}/DCUF_code")
    db_path = "data/feverous-wiki-docs.db"
    doc_db = DocDB(db_path)
    main(args, doc_db)
import argparse
import json
from tqdm import tqdm
from utils.annotation_processor import AnnotationProcessor, EvidenceType

import unicodedata
from cleantext import clean
from urllib.parse import unquote


def average(list):
    return float(sum(list) / len(list))


def clean_title(text):
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--max_page', type=int, default=5)
    parser.add_argument('--max_sent', type=int, default=5)
    parser.add_argument('--all', type=int, default=1)
    parser.add_argument('--input_path', type=str)
    args = parser.parse_args()

    in_path = '{0}/{1}.sentences.roberta.p{2}.s{3}.jsonl'.format(args.input_path, args.split, args.max_page, args.max_sent)

    coverage = []
    precision = []
    coverage_all = []
    precision_all = []
    annotation_processor = AnnotationProcessor('{0}/{1}.jsonl'.format(args.input_path, args.split))
    if args.all == 0:
        annotation_by_id = {el.get_id(): el for el in annotation_processor if el.has_evidence() and el.get_evidence_type(flat=True) == EvidenceType.SENTENCE}
    else:
        annotation_by_id = {el.get_id(): el for el in annotation_processor if el.has_evidence()}

    with open(in_path,"r") as f:
        for idx,line in enumerate(f):
            
            js = json.loads(line)
            if idx == 0:
                continue
            id = js['id']

            anno = annotation_by_id[id]
            docs_gold = list(set([t for t in anno.get_evidence(flat=True) if '_sentence_' in t]))
            if len(docs_gold) == 0:
                continue
            # print(docs_gold)
            docs_predicted = [item for item in js['predicted_sentences'] if '_sentence_' in item]
            #docs_predicted =  [t[0] + '_' + t[1].replace('s_', 'sentence_') for t in js['predicted_sentences'][:5]]
            #docs_predicted = js['predicted_sentences']
            #docs_predicted = [t[0] + '_' +t[1] for t in js['predicted_evidence'] if 'sentence_' in t[1]]
            # docs_predicted =  [t[0] + '_' + t[1].replace('s_', 'sentence_') for t in js['predicted_sentences']]
            if anno.get_verdict() in ['SUPPORTS', 'REFUTES']:
                coverage_ele = len(set(docs_predicted) & set(docs_gold)) / len(docs_gold)
                if docs_predicted:
                    precision_ele = len(set(docs_predicted) & set(docs_gold)) / len(docs_predicted)
                else:
                    precision_ele = 0
                coverage.append(coverage_ele)
                coverage_all.append(coverage_ele)
                precision.append(precision_ele)
                precision_all.append(precision_ele)
            else:
                coverage_ele = len(set(docs_predicted) & set(docs_gold)) / len(docs_gold)
                if docs_predicted:
                    precision_ele = len(set(docs_predicted) & set(docs_gold)) / len(docs_predicted)
                else:
                    precision_ele = 0
                coverage_all.append(coverage_ele)
                precision_all.append(precision_ele)
    print('Recall:')
    print(average(coverage))
    print(average(coverage_all))
    print('Precision:')
    print(average(precision))
    print(average(precision_all))


import sys
sys.path.append('/home/hunan/feverous/mycode/src')
import argparse
import json
from tqdm import tqdm
from utils.annotation_processor import AnnotationProcessor, EvidenceType
import jsonlines
import os


def average(list):
    return float(sum(list) / len(list))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--max_page', type=int, default=5)
    parser.add_argument('--max_sent', type=int, default=5)
    parser.add_argument('--max_tabs', type=int, default=3)
    parser.add_argument('--data_path', type=str, default = '/home/hunan/feverous/mycode/data')
    args = parser.parse_args()
    split = args.split


    in_path_sent = '{0}/{1}.sentences.roberta.p{2}.s{3}.jsonl'.format(args.data_path, split, args.max_page, args.max_sent)
    sentences_pred = {}
    with open(in_path_sent,"r") as f:
        for idx,line in enumerate(f):
            
            js = json.loads(line)
            if 'predicted_sentences' not in js.keys():
                sentences_pred[js['id']] = []
                continue
            sentences_pred[js['id']] = js['predicted_sentences'][:5]

    in_path_tabs = '{0}/{1}.tables.not_precomputed.p{2}.t{3}.jsonl'.format(args.data_path, split, args.max_page, args.max_tabs)
    tabs_pred = {}
    with open(in_path_tabs,"r") as f:
        for idx,line in enumerate(f):
            js = json.loads(line)
            #tabs_pred[idx] = js['predicted_tables']
            #print('*' * 10)
            #print('Using Index as ID for test set, careful!')
            tabs_pred[js['id']] = js['predicted_tables']

    out_path = '{0}/{1}.roberta_sent.table.not_precomputed.p{2}.s{3}.t{4}.jsonl'.format(args.data_path, split, args.max_page, args.max_sent, args.max_tabs)
    with jsonlines.open(os.path.join(out_path), 'w') as writer:
        with jsonlines.open("{0}/{1}.jsonl".format(args.data_path, args.split)) as f:
            for i,line in enumerate(f.iter()):
                if i == 0:
                    writer.write({'header': ''})
                    continue # skip header line
                 #if len(line['evidence'][0]['content']) == 0: continue
                if line['id'] not in tabs_pred:
                    predicted_tables = []
                else:
                    predicted_tables = tabs_pred[line['id']]
                line['predicted_evidence'] = sentences_pred[line['id']] + predicted_tables
                
                writer.write(line)

import sys
import numpy as np

from my_utils.common_utils import load_pkl_data
import argparse
import jsonlines
from tqdm import tqdm
from collections import defaultdict
from utils.annotation_processor import AnnotationProcessor
def average(list):
    return float(sum(list) / len(list))

def select_cells(cell_score, threshold = 0.1):
    predicted_cells = [item[0] for item in cell_score if item[1] > threshold]
    if len(predicted_cells) > 25:
        cell_score.sort(key = lambda x : float(x[1]), reverse = True)
        predicted_cells = list(list(zip(*cell_score))[0][:25])
    return predicted_cells
def select_sents(sent_score, threshold = 0.1):
    predicted_sents = [item[0] for item in sent_score if np.exp(item[1][1]) > threshold]
    return predicted_sents

def get_metric_item(docs_gold, docs_predicted):
    coverage_ele = len(set(docs_predicted) & set(docs_gold)) / len(docs_gold)
    if docs_predicted:
        precision_ele = len(set(docs_predicted) & set(docs_gold)) / len(docs_predicted)
    else:
        precision_ele = 0
    return coverage_ele, precision_ele

def get_evidence_type(gold_evidence):
    ## 0 for sentence
    ## 1 for cell
    ## 2 for joint
    type_set = set()
    for evi in gold_evidence:
        if '_cell_' in evi:
            type_set.add(1)
        elif '_sentence' in evi:
            type_set.add(0)
    if len(type_set) > 1:
        return 2
    
    else:
        return type_set.pop()

def display_follow_evidence_type(data_dict):
    type_dict = {0 : 'SENTENCE', 1 : 'TABLE', 2 : 'JOINT'}
    for key in data_dict.keys():
        print(type_dict[key], end = ':')
        print(average(data_dict[key]))
    print('\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type = str, default= 'dev')
    parser.add_argument('--cell_threshold', type = float, default = 0.0)
    parser.add_argument('--sent_threshold', type = float, default= 0.0)
    args = parser.parse_args()
    print(args)
    raw_data = load_pkl_data('data/{1}.allcells.scores.jsonl'.format(args.input_path, args.split))

    input_file = '{0}/{1}.combined.not_precomputed.p5.s5.t3.jsonl'.format(args.input_path, args.split)

    output_file = '{0}/{1}.fusion.results.jsonl'.format(args.input_path, args.split)

    annotation_processor = AnnotationProcessor('{0}/{1}.jsonl'.format(args.input_path, args.split))
    

    if args.split == 'test':
        annotation_by_id = {el.get_id(): el for el in annotation_processor}
    else:
        annotation_by_id = {el.get_id(): el for el in annotation_processor if el.has_evidence()}

    sent_selection_ratio_by_type = defaultdict(list)
    cell_selection_ratio_by_type = defaultdict(list)
    sent_selection_num_by_type = defaultdict(list)
    cell_selection_num_by_type = defaultdict(list)
    sentence_evidence_ratio_by_type = defaultdict(list)

    coverage_sents = []
    coverage_cells = []
    precision_sents = []
    precision_cells = []
    with jsonlines.open(input_file, 'r') as f:
        with jsonlines.open(output_file, 'w') as writer:
            for idx, line in tqdm(enumerate(f)):
                if idx == 0:
                    writer.write({'header':''})
                    continue
                evi_id = line['id']
                if evi_id in raw_data.keys():
                    cell_score, sent_score = raw_data[evi_id]
                    predicted_sents = select_sents(sent_score, threshold= args.sent_threshold)
                    predicted_cells = select_cells(cell_score, threshold= args.cell_threshold)
                    line['predicted_evidence'] = predicted_sents + predicted_cells
                else:
                    line['predicted_evidence'] = [item[0] + '_' + item[1] for item in line['predicted_evidence']]
                    predicted_sents = [item for item in line['predicted_evidence'] if '_sentence_' in item]
                    predicted_cells = []
                if args.split != 'test':
                    sents_gold = set()
                    cells_gold = set()
                    
                    for item in annotation_by_id[evi_id].get_evidence(flat = True):
                        if '_sentence_' in item:
                            sents_gold.add(item)
                        elif '_cell_' in item:
                            cells_gold.add(item)
                    if evi_id in raw_data.keys() and (len(sents_gold) + len(cells_gold)) > 0:
                        evidence_type = get_evidence_type(sents_gold.union(cells_gold))
                        sent_selection_ratio_by_type[evidence_type].append(len(predicted_sents) / len(sent_score))
                        cell_selection_ratio_by_type[evidence_type].append(len(predicted_cells) / len(cell_score))
                        sent_selection_num_by_type[evidence_type].append(len(predicted_sents))
                        cell_selection_num_by_type[evidence_type].append(len(predicted_cells))
                        if (len(predicted_cells) + len(predicted_sents)) > 0:
                            sentence_evidence_ratio_by_type[evidence_type].append(len(predicted_sents) / (len(predicted_cells) + len(predicted_sents)))
                    if sents_gold:
                        sent_coverage_ele, sent_precision_ele = get_metric_item(sents_gold, predicted_sents)
                        coverage_sents.append(sent_coverage_ele)
                        precision_sents.append(sent_precision_ele)
                    if cells_gold:
                        cell_coverage_ele, cell_precision_ele = get_metric_item(cells_gold, predicted_cells)
                        coverage_cells.append(cell_coverage_ele)
                        precision_cells.append(cell_precision_ele)
                
                writer.write(line)
    if args.split != 'test':
        print("Cell Recall")
        print(average(coverage_cells))
        print('Cell Precision')
        print(average(precision_cells))

        print('Sentence Recall')
        print(average(coverage_sents))
        print('Sentence Precision')
        print(average(precision_sents))

        print('Average Statistics on Evidence Type')
        print('Sentence Selection Number')
        display_follow_evidence_type(sent_selection_num_by_type)
        print('Sentence Selection Ratio From Candiates (top5)')
        display_follow_evidence_type(sent_selection_ratio_by_type)
        print('Cell Selection Number')
        display_follow_evidence_type(cell_selection_num_by_type)
        print('Cell Selection Ratio From Candiates (all cells in all tables)')
        display_follow_evidence_type(cell_selection_ratio_by_type)
        print('Sentence Evidence Ratio in Retrieved Evidence')
        display_follow_evidence_type(sentence_evidence_ratio_by_type)
    print('Retrieval Finished')
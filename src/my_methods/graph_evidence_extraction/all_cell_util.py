import pandas as pd
from collections import defaultdict
from my_utils.common_utils import load_pkl_data, save_pkl_data
import pickle
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModel, TapasTokenizer
import argparse
from tqdm import tqdm
import os
import re
import torch
import random
from utils.annotation_processor import AnnotationProcessor, EvidenceType
from utils.wiki_page import WikiPage, get_wikipage_by_id,WikiTable
from database.feverous_db import FeverousDB
from utils.log_helper import LogHelper
from fusion_col_parser import get_parser
import os
import dgl
import stanza
from fusion_col_util import check_cell_graph, clean_hyperlink_brakets, build_edges_based_on_entity, check_cell_graph, show_data_for_debugging

def prepare_table_with_entity(curr_tab, gold_evidence):
    page = curr_tab.page
    table_content_2D = []
    table_ids_2D = []
    output_candidates = []
    candidates_content = []
    col_labels = set()

    for i, row in enumerate(curr_tab.rows):
        if i == 256 or len(output_candidates) > 512:
            break
        row_id = []
        row_flat = []
        
        for j, cell in enumerate(row.row):
            if j == 128:
                break
            curr_id =page + '_' + cell.get_id()# get current cell id
            if curr_id in gold_evidence:
                col_labels.add(j)
            row_id.append(curr_id)# row id list
            row_flat.append(str(cell))# row content list
            if curr_id not in output_candidates and str(cell):
                output_candidates.append(curr_id)
                candidates_content.append(str(cell))

        table_ids_2D.append(row_id)
        table_content_2D.append(row_flat)
    
    col_labels = [1 if i in col_labels else 0 for i in range(len(table_ids_2D[0]))]
    table = pd.DataFrame(table_content_2D, columns = table_content_2D[0], dtype = str).fillna('')
    return table_ids_2D, table, col_labels, output_candidates, candidates_content

def is_header_row(row):
    ##input row: a list of ids
    return len([ele for ele in row if 'header' not in ele]) == 0

def connect_cell_inset(edge_set, node_set, id_sequence):
    for i in node_set:#i is cell_id
        if i not in id_sequence:
            continue
        for j in node_set:
            if j not in id_sequence:
                continue
            edge_set.add((id_sequence.index(i), id_sequence.index(j)))
    return edge_set, set()

def connect_table(output_ids, id_sequence, connect_pattern = 'left_side_header'):
    assert(output_ids[0] != 'claim')
    assert(connect_pattern in ['left_side_header','both_side_headers']), 'Graph Connecting Pattern not Defined'
    #edge_set = defaultdict(set)
    id_sequence = [item for evi in id_sequence for item in evi]
    edge_set = set()
    ###connect elements in the same row
    for row in output_ids:
        cell_id_set = set()
        if is_header_row(row):
            for cell_id in row:
                cell_id_set.add(cell_id)
            edge_set, cell_id_set = connect_cell_inset(edge_set, cell_id_set, id_sequence)

        else:
            head_ids = set()
            for i, cell_id in enumerate(row):
                if 'header' in cell_id:
                    head_ids.add(cell_id)

                if ('header' in cell_id or i == len(row)-1) and i != 0:
                    if connect_pattern == 'both_side_headers' or i == len(row)-1:
                        cell_id_set.add(cell_id)
                    edge_set, cell_id_set = connect_cell_inset(edge_set, cell_id_set, id_sequence)
                cell_id_set.add(cell_id)
            edge_set, head_ids = connect_cell_inset(edge_set, head_ids, id_sequence)

    ###connect elements in the same column
    output_ids = list(zip(*output_ids))
    for row in output_ids:
        cell_id_set = set()
        if is_header_row(row):
            for cell_id in row:
                cell_id_set.add(cell_id)
            edge_set, cell_id_set = connect_cell_inset(edge_set, cell_id_set, id_sequence)

        else:
            head_ids = set()
            for i, cell_id in enumerate(row):
                if 'header' in cell_id:
                    head_ids.add(cell_id)

                if ('header' in cell_id or i == len(row)-1) and i != 0:
                    if connect_pattern == 'both_side_headers' or i == len(row)-1:
                        cell_id_set.add(cell_id)
                    edge_set, cell_id_set = connect_cell_inset(edge_set, cell_id_set, id_sequence)
                cell_id_set.add(cell_id)
            edge_set, head_ids = connect_cell_inset(edge_set, head_ids, id_sequence)

    return edge_set

def build_data_with_entity_edges(anno, nlp, test_mode, args):
    ### claim for roberta and tapas
    ### sentence for roberta
    ### table for tapas
    ### table_ids_2d for pooling
    ### edge_dict
    ### sentence_id
    ### cell_id
    ### sentence labels
    ### col labels
    ### cell labels
    id = anno.id
    claim = anno.claim
    sentence_ids = []
    table_ids = set()
    cell_ids = []
    table_content = []
    table_ids_2D = []
    if test_mode:
        gold_evidence = []
    else:
        gold_evidence = anno.get_evidence(flat = True)
    for evi in anno.predicted_evidence:
        if '_sentence_' in evi:
            sentence_ids.append(evi)
        elif 'table_' in evi[1]:
            table_ids.add('_'.join(evi))
    if not sentence_ids or not table_ids:
        return [None] * 12
    sentences = []
    entity_pool = defaultdict(set)
    hyperlink_pool = defaultdict(set)
    sentence_ids = list(set(sentence_ids))
    for evi in sentence_ids:
        page_id = evi.split('_')[0]
        sent_id = '_'.join(evi.split('_')[1:])
        page_json = args.db.get_doc_json(page_id)
        curr_page = WikiPage(page_id, page_json)
        if curr_page is None:
            continue
        hyperlink, sentence = clean_hyperlink_brakets(curr_page.get_element_by_id(sent_id).content)
        for item in hyperlink:
            hyperlink_pool[item].add(evi)
        sentences.append(sentence)
    if sentences:
        in_docs = [stanza.Document([], text=d) for d in sentences]
        for i, sent in enumerate(nlp(in_docs)):
            if sent.sentences:
                for token in sent.sentences[0].tokens:
                    if token.ner != 'O':
                        entity_pool[token.text].add(sentence_ids[i])
        
    col_labels = []
    for evi in table_ids:
        page_id = evi.split('_')[0]
        table_id = '_'.join(evi.split('_')[1:])
        page_json = args.db.get_doc_json(page_id)
        curr_page = WikiPage(page_id, page_json)
        if curr_page is None:
            continue
        curr_table = curr_page.get_element_by_id(table_id)

        curr_id_2D, curr_content, curr_col_labels, curr_cell_ids, curr_cell_content = prepare_table_with_entity(curr_table, gold_evidence)
        for i, content in enumerate(curr_cell_content):
            hyperlink, content = clean_hyperlink_brakets(content)
            for item in hyperlink:
                hyperlink_pool[item].add(curr_cell_ids[i])
            curr_cell_content[i] = content
        if curr_cell_content:
            in_docs = [stanza.Document([], text=d) for d in curr_cell_content]
            for i, sent in enumerate(nlp(in_docs)):
                if sent.sentences:
                    for token in sent.sentences[0].tokens:
                        if token.ner != 'O':
                            entity_pool[token.text].add(curr_cell_ids[i])
        table_content.append(curr_content)
        table_ids_2D.append(curr_id_2D)
        col_labels.append(curr_col_labels)
        cell_ids.append(curr_cell_ids)

    cell_table_edges = set()
    for curr_table_ids_2D in table_ids_2D:
        cell_table_edges = cell_table_edges.union(connect_table(curr_table_ids_2D, cell_ids))

    sent_col_edges, cell_entity_edges = build_edges_based_on_entity(sentence_ids, cell_ids, entity_pool, hyperlink_pool, table_ids, table_ids_2D, cell2cell_page= False)
    
    cell_edges = cell_table_edges.union(cell_entity_edges)
    sentence_labels = [1 if evi in gold_evidence else 0 for evi in sentence_ids]
    cell_labels = [1 if evi in gold_evidence else 0  for item in cell_ids for evi in item]
    col_labels = [evi_label for item in col_labels for evi_label in item]
    return id, claim, sentences, table_content, table_ids_2D, sentence_ids, cell_ids,sentence_labels, col_labels, cell_labels, sent_col_edges, cell_edges


def build_dataset_with_all_cells(split, args, cache_file = None, test_mode = False):
    
    if os.path.exists(cache_file):
        print('Data Loading From ' + cache_file +' ...')
        output = load_pkl_data(cache_file)
        return output

    anno_file = args.input_path + '/{}.roberta.not_precomputed.p5.s5.t3.jsonl'.format(split)
    output = defaultdict(list)
    anno_processor = AnnotationProcessor(anno_file)
    annotations = [anno for anno in anno_processor]
    nlp = stanza.Pipeline('en', processors="tokenize,ner", verbose=False, tokenize_pretokenized=True, download_method= None)
    for i, anno in tqdm(enumerate(annotations), desc = 'build dataset at ' + cache_file):    
    ### claim for roberta and tapas
    ### sentence for roberta
    ### table for tapas
    ### table_ids_2d for pooling
    ### edge_dict
    ### sentence_id
    ### cell_id
    ### sentence labels
    ### col labels
    ### cell labels

    ### filter
    ### filter out data only with retrieved sentence or tables
    ### Use data containing both format of data in training.
    ### the ids are collected. These can be used to construct graphs afterwards
        id, claim, sentences, table_content, table_ids_2D, sentence_ids, cell_ids, sentence_labels, col_labels, cell_labels, sent_col_edges, cell_edges = build_data_with_entity_edges(anno, nlp, test_mode, args)
        if not sentences or not table_content or not cell_labels:
            continue

        output['id'].append(id)
        output['claim'].append(claim)
        output['sentences'].append(sentences)
        output['tables'].append(table_content)
        output['table_ids_2D'].append(table_ids_2D)
        output['sent_ids'].append(sentence_ids)
        output['cell_ids'].append(cell_ids)
        output['sent_labels'].append(sentence_labels)
        output['col_labels'].append(col_labels)
        output['cell_labels'].append(cell_labels)
        output['sent_col_edges'].append(sent_col_edges)
        output['cell_edges'].append(cell_edges)
       
    if cache_file:
        save_pkl_data(output, cache_file)

    return output

if __name__ == '__main__':
    args = get_parser().parse_args()
    args.db = FeverousDB(args.wiki_path)

#    output = build_dataset_with_all_cells('dev', args, cache_file = '/home/hunan/feverous/mycode/src/my_methods/fusion_graph_with_col/cache_data/dev_all_cells.jsonl')
 #   check_cell_graph(output)
    output = build_dataset_with_all_cells('test', args, cache_file = '/home/hunan/feverous/mycode/src/my_methods/fusion_graph_with_col/cache_data/test_all_cells.jsonl', test_mode = True)

    check_cell_graph(output)

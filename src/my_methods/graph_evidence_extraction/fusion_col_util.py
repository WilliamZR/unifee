from matplotlib.pyplot import annotate
import pandas as pd
from collections import defaultdict
from my_utils.common_utils import load_pkl_data, save_pkl_data
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModel, TapasTokenizer
import argparse
from tqdm import tqdm
import os
import re
import torch
from utils.annotation_processor import AnnotationProcessor, EvidenceType
from utils.wiki_page import WikiPage, get_wikipage_by_id,WikiTable
from database.feverous_db import FeverousDB
from fusion_col_parser import get_parser
import os
import dgl
import stanza

def connect_pooling_matrix(pooling_matrix):
    #cellids * 512
    if len(pooling_matrix) == 1:
        return pooling_matrix[0]
    left = pooling_matrix.pop(0)
    right = pooling_matrix[0]
    left_temp = torch.cat((left, torch.zeros(left.size()[0], right.size()[1])), dim = 1)
    right_temp = torch.cat((torch.zeros(right.size()[0], left.size()[1]), right), dim = 1) 
    pooling_matrix[0] = torch.cat((left_temp, right_temp))
    return connect_pooling_matrix(pooling_matrix)

def collate_fn(batch):
    sent_input, sent_mask\
        , table_input_ids, table_attention_masks, table_token_type_ids\
        , fusion_graph, cell_graph\
        , col_pooling_matrix, pooling_matrix\
        , sent_labels, col_labels, cell_labels = map(list, zip(*batch))

    batch_sent_input = torch.cat(sent_input)
    batch_sent_mask = torch.cat(sent_mask)

    table_input_ids = [torch.reshape(item, (-1, 512)) for item in table_input_ids if item != []]
    table_attention_masks = [torch.reshape(item, (-1, 512)) for item in table_attention_masks if item != []]
    table_token_type_ids = [torch.reshape(item, (-1, 512, 7)) for item in table_token_type_ids if item != []]
    if table_input_ids != []:
        batch_table_ids = torch.cat(table_input_ids)
        batch_table_attention_masks = torch.cat(table_attention_masks)
        batch_table_token_type_ids = torch.cat(table_token_type_ids)

        pooling_matrix = [item for item in pooling_matrix if item != []]
        batch_pooling_matrix = connect_pooling_matrix(pooling_matrix)
        col_pooling_matrix = [item for item in col_pooling_matrix if item != []]
        batch_col_pooling_matrix = connect_pooling_matrix(col_pooling_matrix)
    else:
        batch_table_ids = torch.tensor([])
        batch_table_attention_masks = torch.tensor([])
        batch_table_token_type_ids = torch.tensor([])
        batch_pooling_matrix = torch.tensor([])
        batch_col_pooling_matrix = torch.tensor([])

    if len(fusion_graph) == 1:
        batch_graph = fusion_graph[0]
        batch_cell_graph = cell_graph[0]
    else:
        batch_graph = dgl.batch(fusion_graph)
        batch_cell_graph = dgl.batch(cell_graph)

    batch_sent_labels = torch.tensor([i for item in sent_labels for i in item])
    batch_cell_labels = torch.tensor([i for item in cell_labels for i in item])
    batch_col_labels = torch.tensor([i for item in col_labels for i in item])

    return batch_sent_input, batch_sent_mask\
        , batch_table_ids, batch_table_attention_masks, batch_table_token_type_ids\
        , batch_graph, batch_cell_graph\
        , batch_col_pooling_matrix, batch_pooling_matrix\
        , batch_sent_labels, batch_col_labels, batch_cell_labels

class FusionRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(args.roberta_path)
        self.tapas_tokenizer = AutoTokenizer.from_pretrained(args.tapas_path)
        self.use_entity = args.use_entity_edges

    def __getitem__(self, idx):
        ### claim for roberta and tapas
    ### sentence for roberta
    ### table for tapas
    ### table_ids_2d for pooling
    ### sentence_id
    ### cell_id
    ### sentence labels
    ### cell labels
        claim = self.data['claim'][idx]
        sentences = self.data['sentences'][idx]
        tables = self.data['tables'][idx]
        table_ids_2D = self.data['table_ids_2D'][idx]
        cell_ids = self.data['cell_ids'][idx]
        sent_ids = self.data['sent_ids'][idx]
        sent_labels = self.data['sent_labels'][idx]
        col_labels = self.data['col_labels'][idx]
        cell_labels = self.data['cell_labels'][idx]

        claims = [claim]* len(sentences)

        sentence_input = self.roberta_tokenizer(claims, sentences, padding = 'max_length', max_length = 512, truncation = True, return_tensors = 'pt')
        
        table_input_ids = []
        table_attention_masks = []
        table_token_type_ids = []
        pooling_matrix = []
        col_pooling_matrix = []
        if len(cell_ids) > 0:
            for i in range(len(cell_ids)):
                temp_input, temp_mask, temp_token_type, temp_pooling_matrix, temp_col_pooling_matrix = self.tokenize_claim_and_inputs(claim, tables[i], cell_ids[i], table_ids_2D[i], self.tapas_tokenizer)
                table_input_ids.append(temp_input)
                table_attention_masks.append(temp_mask)
                table_token_type_ids.append(temp_token_type)
                pooling_matrix.append(temp_pooling_matrix)
                col_pooling_matrix.append(temp_col_pooling_matrix)
            table_input_ids = torch.stack(table_input_ids)
            table_attention_masks = torch.stack(table_attention_masks)
            table_token_type_ids = torch.stack(table_token_type_ids)
            col_pooling_matrix = connect_pooling_matrix(col_pooling_matrix)
            pooling_matrix = connect_pooling_matrix(pooling_matrix)

        # let's try fully connected graphs first
        if self.use_entity:
            fusion_graph_edges = list(self.data['sent_col_edges'][idx])
        else:
            fusion_graph_edges = connect_evi_in_two_sets(sent_ids + col_labels, sent_ids + col_labels)
            fusion_graph_edges = list(fusion_graph_edges)
        fusion_graph = dgl.graph(fusion_graph_edges)
        
        if self.use_entity:
            cell_graph_edges = list(self.data['cell_edges'][idx])
        else:
            cell_graph_edges = connect_evi_in_two_sets(cell_labels, cell_labels)
            cell_graph_edges = list(cell_graph_edges)
        cell_graph = dgl.graph(cell_graph_edges)

        fusion_graph = dgl.add_self_loop(fusion_graph)
        graph = dgl.add_self_loop(graph)
        if col_pooling_matrix != []:
            graph.ndata['t'] = torch.tensor([0] * len(sentences) + [1] * col_pooling_matrix.size()[0])
        else:
            graph.ndata['t'] = torch.tensor([0] * len(sentences))
        cell_graph = dgl.add_self_loop(cell_graph)
        return sentence_input['input_ids'], sentence_input['attention_mask']\
            , table_input_ids, table_attention_masks, table_token_type_ids\
                , fusion_graph, cell_graph\
                , col_pooling_matrix, pooling_matrix\
                , sent_labels, col_labels, cell_labels

    def __len__(self):
        return len(self.data['claim'])

    def tokenize_claim_and_inputs(self, claim, table, id_sequence, table_ids_2D, tokenizer):
        tokenized_inputs = tokenizer(table = table, queries = claim, padding = 'max_length', truncation = True, return_tensors = 'pt')
        pooling_matrix = torch.zeros([len(id_sequence), 512])
        col_pooling_matrix = torch.zeros([len(table_ids_2D[0]), 512])

        for i in range(512):
            if tokenized_inputs['token_type_ids'][0, i, 0] == 0:
                continue
            elif tokenized_inputs['token_type_ids'][0, i, 0] == 1 and  tokenized_inputs['token_type_ids'][0, i, 1] *  tokenized_inputs['token_type_ids'][0, i, 2] > 0:
                col_num = tokenized_inputs['token_type_ids'][0, i , 1] - 1
                col_pooling_matrix[col_num, i] = 1
        col_pooling_matrix = avearge_of_matrix(col_pooling_matrix)
        ##col_pooling_matrix = col_pooling_matrix
        
        for i in range(512):
            if tokenized_inputs['token_type_ids'][0, i, 0] == 0:
                continue
            elif  tokenized_inputs['token_type_ids'][0, i, 0] == 1 and  tokenized_inputs['token_type_ids'][0, i, 1] *  tokenized_inputs['token_type_ids'][0, i, 2] > 0:
                cell_id = table_ids_2D[tokenized_inputs['token_type_ids'][0, i, 2]-1][tokenized_inputs['token_type_ids'][0, i, 1]-1]
                if cell_id in id_sequence:
                    pooling_matrix[id_sequence.index(cell_id),i] = 1
        pooling_matrix = avearge_of_matrix(pooling_matrix)
        #average_num = torch.sum(pooling_matrix, dim = 1)
        #average_num = torch.reshape(average_num, (-1, 1))
        #average_num = torch.where(average_num == 0, torch.ones_like(average_num) , average_num)
        #pooling_matrix = torch.div(pooling_matrix, average_num)
        return tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], tokenized_inputs['token_type_ids'], pooling_matrix, col_pooling_matrix

    def collate_fn(cls):
        return collate_fn

def avearge_of_matrix(pooling_matrix):
    average_num = torch.sum(pooling_matrix, dim = 1)
    average_num = torch.reshape(average_num, (-1, 1))
    average_num = torch.where(average_num == 0, torch.ones_like(average_num), average_num)
    pooling_matrix = torch.div(pooling_matrix, average_num)
    return pooling_matrix

def connect_evi_in_two_sets(id_sequence1, id_sequence2):
    output_edge_set = set()
    for i in range(len(id_sequence1)):
        for j in range(len(id_sequence2)):
            output_edge_set.add((i, j))

    return output_edge_set 



def build_edge_dict(sent_ids, cell_ids_2d):
    cell_ids = []
    for item in cell_ids_2d:
        cell_ids += item
    edge_dict = {}
    edge_dict[('sentence', 'sent2sent', 'sentence')] = connect_evi_in_two_sets(sent_ids, sent_ids)
    edge_dict[('sentence', 'sent2cell', 'cell')] = connect_evi_in_two_sets(sent_ids, cell_ids)
    edge_dict[('cell', 'sent2sent', 'sentence')] = connect_evi_in_two_sets(cell_ids, sent_ids)
    edge_dict[('cell', 'sent2sent', 'cell')] = connect_evi_in_two_sets(cell_ids, cell_ids)
    return edge_dict

def prepare_table(curr_tab, gold_evidence):
    page = curr_tab.page
    table_content_2D = []
    table_ids_2D = []
    col_labels = set()
    id_sequence = []
    for i, row in enumerate(curr_tab.rows):
        if i == 256:
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

        table_ids_2D.append(row_id)
        table_content_2D.append(row_flat)
    
    col_labels = [1 if i in col_labels else 0 for i in range(len(table_ids_2D[0]))]
    table = pd.DataFrame(table_content_2D, columns=table_content_2D[0], dtype = str).fillna('')
    return table_ids_2D, table, col_labels

def clean_hyperlink_brakets(sentence):
    hyperlink = re.findall('\[\[.*?\|', sentence)
    hyperlink = [item[2:-1] for item in hyperlink]
    sentence = re.sub('\[\[.*?\|', '', sentence)
    sentence = re.sub('\]\]', '', sentence)
    return hyperlink, sentence

def build_dataset_batch(anno, args):
    ### This function is already checked
    ### This is the old function for fully connected graph
    ### return:
    ### id
    ### claim for roberta and tapas
    ### sentence for roberta
    ### table for tapas
    ### table_ids_2d for pooling
    ### edge_dict
    ### sentence_id
    ### cell_id
    ### sentence labels
    ### col_labes
    ### cell labels
    id = anno.id
    claim = anno.claim
    sentence_ids = []
    table_ids = set()
    cell_ids = []
    table_content = []
    table_ids_2D = []
    cell_id_by_table = defaultdict(list)
    gold_evidence = anno.get_evidence(flat = True)
    for evi in anno.predicted_evidence:
        if '_sentence_' in evi:
            sentence_ids.append(evi)
        elif '_table_' in evi:
            table_ids.add(evi)
        elif '_cell_' in evi:
            table_id = evi.split('_')[0] + '_table_' + evi.split('_')[-3]
            if table_id in table_ids:
                cell_id_by_table[table_id].append(evi)
    sentences = []
    for evi in sentence_ids:
        page_id = evi.split('_')[0]
        sent_id = '_'.join(evi.split('_')[1:])
        page_json = args.db.get_doc_json(page_id)
        curr_page = WikiPage(page_id, page_json)
        if curr_page is None:
            continue
        sentences.append(curr_page.get_element_by_id(sent_id).content)

    col_labels = []
    for evi in table_ids:
        cell_ids.append(cell_id_by_table[evi])
        page_id = evi.split('_')[0]
        table_id = '_'.join(evi.split('_')[1:])
        page_json = args.db.get_doc_json(page_id)
        curr_page = WikiPage(page_id, page_json)
        if curr_page is None:
            continue
        curr_table = curr_page.get_element_by_id(table_id)
        curr_id_2D, curr_content, curr_col_labels = prepare_table(curr_table, gold_evidence)
        table_content.append(curr_content)
        table_ids_2D.append(curr_id_2D)
        col_labels += curr_col_labels
    
    sentence_labels = [1 if evi in gold_evidence else 0 for evi in sentence_ids]
    cell_labels = [1 if evi in gold_evidence else 0  for item in cell_ids for evi in item]

    #edge_dict = build_edge_dict(sentence_ids, cell_ids)

    return id, claim, sentences, table_content, table_ids_2D, sentence_ids, cell_ids, sentence_labels, col_labels, cell_labels

def prepare_table_with_entity(curr_tab, gold_evidence, candidate_ids):
    page = curr_tab.page
    table_content_2D = []
    table_ids_2D = []
    col_labels = set()
    output = set()
    output_candidates = []
    candidates_content = []
    for i, row in enumerate(curr_tab.rows):
        if i == 256:
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
            if curr_id in candidate_ids and curr_id not in output_candidates:
                output_candidates.append(curr_id)
                candidates_content.append(str(cell))

        table_ids_2D.append(row_id)
        table_content_2D.append(row_flat)

    col_labels = [1 if i in col_labels else 0 for i in range(len(table_ids_2D[0]))]
    table = pd.DataFrame(table_content_2D, columns=table_content_2D[0], dtype = str).fillna('')
    return table_ids_2D, table, col_labels, output_candidates, list(candidates_content)

def connect_in_one_set(evi_set, evi_ids):
    edge_set = set()
    for i in evi_set:
        for j in evi_set:
            edge_set.add((evi_ids.index(i),evi_ids.index(j)))
    return edge_set
        
def connect_evidence_in_pool(evidence_pool, sent_col_ids, cell_ids):
    sent_col_edges = set()
    cell_edges = set()
    sent_pool = set()
    cell_pool = set()
    for evi in evidence_pool:
        if '_sentence_' in evi:
            sent_pool.add(evi)
        if '_cell_' in evi:
            col_id_temp = evi.split('_')
            col_id = col_id_temp[0] + '_table_' + col_id_temp[-3] + '_col_' + col_id_temp[-1] 
            sent_pool.add(col_id)
            cell_pool.add(evi)
    sent_col_edges = sent_col_edges.union(connect_in_one_set(sent_pool, sent_col_ids))
    cell_edges = cell_edges.union(connect_in_one_set(cell_pool, cell_ids))

    return sent_col_edges, cell_edges
    
        
def generate_col_id_sequence(table_ids, table_content_ids_2D):
    col_id_sequence = []
    for i, curr_tab_id in enumerate(table_ids):
        for j in range(len(table_content_ids_2D[i][0])):
            col_id_sequence.append(curr_tab_id +'_col_' + str(j))

    return col_id_sequence


def build_edges_based_on_entity(sentence_ids, cell_ids, entity_pool, hyperlink_pool, table_ids, table_content_ids_2D, cell2cell_page = True):
    ## entity pool: {entity: sentence or cell}
    ## hyperlink: {hyperlink: sentence or cell}
    ## evi_by_pages
    cell_ids = [id for item in cell_ids for id in item]
    sent_col_edges = set()
    cell_edges = set()
    sent_col_ids = sentence_ids + generate_col_id_sequence(table_ids, table_content_ids_2D)
    ### connect cell2cell col/sent2col/sent if in same pool
    ### connect sent with col if sent and its daughter cell share entity/hyperlink

    for key in entity_pool.keys():
        curr_sent_col_edges, curr_cell_edges = connect_evidence_in_pool(entity_pool[key], sent_col_ids, cell_ids)
        sent_col_edges = sent_col_edges.union(curr_sent_col_edges)
        cell_edges = cell_edges.union(curr_cell_edges)

    for key in hyperlink_pool.keys():
        curr_sent_col_edges, curr_cell_edges = connect_evidence_in_pool(hyperlink_pool[key], sent_col_ids, cell_ids)
        sent_col_edges = sent_col_edges.union(curr_sent_col_edges)
        cell_edges = cell_edges.union(curr_cell_edges)

    ### connect cell2cell if in same page
    ### DO NOT use this connection if you use all the cells as candiates!
    if cell2cell_page:
        evi_by_page = defaultdict(set)
        for evi_id in cell_ids:
            page = evi_id.split('_')[0]
            evi_by_page[page].add(evi_id)
        for page in evi_by_page.keys():
            curr_cell_edges = connect_in_one_set(evi_by_page[page], cell_ids)
            cell_edges = cell_edges.union(curr_cell_edges)
    ### connect sent2sent col2col sent2col if in same page
    evi_by_page = defaultdict(set)
    for evi_id in sent_col_ids:
        page = evi_id.split('_')[0]
        evi_by_page[page].add(evi_id)
    for page in evi_by_page.keys():
        curr_sent_col_edges = connect_in_one_set(evi_by_page[page], sent_col_ids)
        sent_col_edges = sent_col_edges.union(curr_sent_col_edges)
    return sent_col_edges, cell_edges


def build_dataset_with_entity_edges(anno, nlp, args):
     ### This function is already checked
    ### return:
    ### id
    ### claim for roberta and tapas
    ### sentence for roberta
    ### table for tapas
    ### table_ids_2d for pooling
    ### edge_dict
    ### sentence_id
    ### cell_id
    ### sentence labels
    ### col_labes
    ### cell labels
    id = anno.id
    claim = anno.claim
    sentence_ids = []
    table_ids = set()
    cell_ids = []
    table_content = []
    table_ids_2D = []
    cell_id_by_table = defaultdict(list)
    gold_evidence = anno.get_evidence(flat = True)
    for evi in anno.predicted_evidence:
        if '_sentence_' in evi:
            sentence_ids.append(evi)
        elif '_table_' in evi:
            table_ids.add(evi)

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
        curr_cell_ids = cell_id_by_table[evi]
        page_id = evi.split('_')[0]
        table_id = '_'.join(evi.split('_')[1:])
        page_json = args.db.get_doc_json(page_id)
        curr_page = WikiPage(page_id, page_json)
        if curr_page is None:
            continue
        curr_table = curr_page.get_element_by_id(table_id)
        curr_id_2D, curr_content, curr_col_labels, curr_cell_ids, curr_cell_content = prepare_table_with_entity(curr_table, gold_evidence, curr_cell_ids)
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
    


    sent_col_edges, cell_edges = build_edges_based_on_entity(sentence_ids, cell_ids, entity_pool, hyperlink_pool, table_ids, table_ids_2D)

    sentence_labels = [1 if evi in gold_evidence else 0 for evi in sentence_ids]
    cell_labels = [1 if evi in gold_evidence else 0  for item in cell_ids for evi in item]
    col_labels = [evi_label for item in col_labels for evi_label in item]

    return id, claim, sentences, table_content, table_ids_2D, sentence_ids, cell_ids, sentence_labels, col_labels, cell_labels, sent_col_edges, cell_edges

def build_dataset(split, args, cache_file = None, train_mode = True, use_entity = True):
    
    if os.path.exists(cache_file):
        print('Data Loading From ' + cache_file +' ...')
        output = load_pkl_data(cache_file)
        return output

    anno_file = '{0}/{1}.roberta_sent.table.not_precomputed.p{2}.s{3}.t{4}.jsonl'.format(args.data_path, split, args.max_page, args.max_sent, args.max_tabs)

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
        
        id, claim, sentences, table_content, table_ids_2D, sentence_ids, cell_ids, sentence_labels, col_labels, cell_labels, sent_col_edges, cell_edges = build_dataset_with_entity_edges(anno, nlp, args)
        if not sentences or not table_content or not cell_labels:
            continue
        if 1 not in sentence_labels and 1 not in cell_labels:
            if train_mode:
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



def check_cell_graph(data):
    ### This function is for debugging
    ### You may encounter this bug during training
    ### dgl._ffi.base.DGLError: There are 0-in-degree nodes in the graph, output for those nodes will be invalid.
    ### Reason could be:
    ##### some ids are repeated in the id_sequence, causing no in-degree and redutant nodes
    ### This function is to localate the bug
    for i in tqdm(range(len(data['id']))):
        if i < 45000:
            continue
        cell_graph_edges = data['cell_edges'][i]
        cell_graph = dgl.graph(list(cell_graph_edges))
        cell_ids = data['cell_ids'][i]
        cell_ids = [item for evi in cell_ids for item in evi]

        if 0 in cell_graph.in_degrees():
            print('Found 0 in degrees at cell graph'+ str(data['id'][i]))
            print('Index ' + str(i))
            show_data_for_debugging(data, i)
        if cell_graph.num_nodes() != len(cell_ids):
            print('num nodes not match')
            print(len(cell_ids))
            print(cell_graph.num_nodes())
            print(data['id'][i])
            print(i)
        


def show_data_for_debugging(data, idx):
    print(data['id'][idx])
    print(data['table_ids_2D'][idx])
    print(data['tables'][idx])
    print(data['cell_ids'][idx])
    print(len(data['cell_labels'][idx]))
    print(data['cell_labels'][idx])
    cell_graph = dgl.graph(list(data['cell_edges'][idx]))
    print(cell_graph.num_nodes())
    print(cell_graph.in_degrees())

if __name__ == '__main__':
    args = get_parser().parse_args()
    args.db = FeverousDB(args.wiki_path)

    output = build_dataset(args.split , args, cache_file='{0}/{1}_entity_graph.jsonl', train_mode= (args.split == 'train'))
    
    

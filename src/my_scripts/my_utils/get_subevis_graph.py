# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/8/30 21:06
# Description:

import numpy as np
from scipy import sparse
import dgl

def get_subevis_graph(entry, args):
    node_num = args.cla_evi_num
    edges = np.zeros([node_num, node_num])

    evidences = entry["evidences"][:node_num-1]
    evidences_title = entry["evidences_title"][:node_num-1]
    evidences_meta = entry["evidences_meta"][:node_num-1]

    while evidences_title and (evidences_title[-1] is None):
        evidences.pop()
        evidences_title.pop()
        evidences_meta.pop()

    true_node_num = min(node_num, 1+len(evidences))

    edges = _add_self_loop(edges, true_node_num)
    edges = _add_meta_overlap(edges, evidences_meta)
    if args.use_evi_mask:
        evidences_mask = entry["evidences_mask"][:node_num - 1]
        edges = _add_cla_evi_masked(edges, true_node_num, evidences_mask)
    else:
        edges = _add_cla_evi(edges, true_node_num)

    sparse_edges = sparse.csr_matrix(edges)
    graph = dgl.from_scipy(sparse_edges)
    # print(edges)
    # print(sparse_edges)
    # print(graph)
    try:
        graph_words = [entry["claim"]]
        graph_words.extend([evi_t + " : " + evi for evi_t, evi in zip(evidences_title, evidences)])
    except:
        a = 1
        assert False
    if len(graph_words) < node_num:
        graph_words += [''] * (node_num - len(graph_words))
    graph_words = graph_words[:node_num]

    return graph, graph_words

def _add_self_loop(edges, node_num):
    for i in range(node_num):
        edges[i][i] = 1
    return edges

def _add_meta_overlap(edges, evidences_meta):
    evidences_meta.insert(0, None)
    for i in range(len(evidences_meta)):
        for j in range(i+1, len(evidences_meta)):
            if evidences_meta[i] == evidences_meta[j]:
                edges[i][j] = edges[j][i] = 1
    return edges

def _add_cla_evi(edges, node_num):
    for i in range(1, node_num):
        edges[0][i] = edges[i][0] = 1
    return edges

def _add_cla_evi_masked(edges, node_num, evi_mask):
    assert node_num-1 == len(evi_mask)
    for i,em in enumerate(evi_mask):
        #单向
        edges[i+1][0] = em
    return edges

# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/1/3 17:52
# Description:
import numpy as np
from deprecated.sphinx import deprecated
from scipy import sparse
import dgl


def _add_edges(lst_edge, word_num, sent_lens, token_lst, entry, args):
    lst_edge = _add_self_loop(lst_edge, word_num)

    if not args.srl_claim:
        lst_edge = _add_fully_claim(lst_edge, sent_lens)

    if args.use_gmn:
        lst_edge = _add_gmn(lst_edge, word_num, sent_lens)
    else:
        lst_edge = _add_overlap(lst_edge, token_lst)
        if args.cluster_graph_words:
            lst_edge = _add_cluster_words(entry, token_lst, lst_edge)
    # lst_edge = _add_adjacent(lst_edge, token_lst) #?
    return lst_edge

def get_dep_graph(dep_dicts, words, sent_lens, entry, args, edge_func = _add_edges):
    edges = []
    token_lst = []
    graph_words = []
    for w1, dep_dict, sent_len in zip(words, dep_dicts,sent_lens):
        # srl切分后的词汇和bert tokenizer切分后的词汇对齐
        w2 = dep_dict["words"]

        assert w1 == w2
        tokens = w2
        word_num = len(tokens)
        assert word_num == sent_len, print(word_num, sent_len)
        token_lst += tokens
        graph_words.append(tokens)
        word_num = len(tokens)
        edge = np.zeros([word_num, word_num])

        deps = dep_dict["dependencies"]["dep"]
        for dep in deps:
            if dep[0] > 0 and dep[1] > 0:
                edge[dep[1] - 1][dep[0] - 1] = 1
                #edge[dep[0] - 1][dep[1] - 1] = 1
        # print(edge)
        edges.append(edge[:sent_len, :sent_len])

    word_num = len(token_lst)
    lst_edge = _merge_edge_lst(edges, word_num)

    lst_edge = edge_func(lst_edge, word_num, sent_lens, token_lst, entry, args)

    sparse_edges = sparse.csr_matrix(lst_edge)
    graph = dgl.from_scipy(sparse_edges)
    # print(edges)
    # print(sparse_edges)
    # print(graph)
    return graph, graph_words

def get_srl_graph(srl_dicts, words, sent_lens, entry, args, edge_func = _add_edges):
    edges = []
    token_lst = []
    graph_words = []
    for w1, srl_dict, sent_len in zip(words, srl_dicts,sent_lens):
        # srl切分后的词汇和bert tokenizer切分后的词汇对齐
        w2 = srl_dict["words"]
        if len(w2) > len(w1):
            for i in range(len(w1)-2):
                if w2[i] == '[' and w2[i+1] == "UNK":
                    w2 = w2[:i] + ["[UNK]"] + w2[i+3:]
                    for verb_entry in srl_dict["verbs"]:
                        verb_entry["tags"] = verb_entry["tags"][:i+1]+verb_entry["tags"][i+3:]

        if len(w2) > len(w1):
            for i in range(len(w1)):
                while w1[i] != w2[i]:
                    w2 = w2[:i]+[w2[i]+w2[i+1]]+w2[i+2:]
                    for verb_entry in srl_dict["verbs"]:
                        verb_entry["tags"] = verb_entry["tags"][:i+1]+verb_entry["tags"][i+2:]

        assert w1 == w2
        tokens = w2
        srl_dict["words"] = tokens

        word_num = len(tokens)
        assert word_num == sent_len, print(word_num, sent_len)
        token_lst += tokens
        graph_words.append(tokens)
        word_num = len(tokens)
        edge = np.zeros([word_num, word_num])

        for idx, wd in enumerate(tokens):
            if wd == ":":
                for i in range(idx):
                    for j in range(i):
                        edge[i][j] = edge[j][i] = 1
                break

        verb_idxs = []
        for verb_entry in srl_dict["verbs"]:
            assert word_num == len(verb_entry["tags"]), print(word_num, '\n', verb_entry["tags"])
            find_verb = False
            for idx, tag in enumerate(verb_entry["tags"]):
                if tag == "B-V":
                    find_verb = True
                    stat = idx

                    edge[stat][0] = 1
                    edge[0][stat] = 1

                    idx += 1
                    while idx < len(verb_entry["tags"]) and verb_entry['tags'][idx] == "I-V":
                        #edge[stat][idx] = 1
                        #edge[idx][stat] = 1
                        idx += 1
                    for i in range(stat, idx):
                        for j in range(i+1, idx):
                            edge[i][j] = 1
                            edge[j][i] = 1
                    verb_idxs.append((stat, idx))
                    break
            if find_verb == False:
                for i, wd in enumerate(srl_dict["words"]):
                    if verb_entry["verb"] == wd:
                        verb_idxs.append((i, i+1))
                        break

        if len(verb_idxs) != len(srl_dict["verbs"]):
            verbs = [' '.join(tokens[verb_idxs[i][0]:verb_idxs[i][1]]) for i in range(len(verb_idxs))]
            s_verbs = [sd for sd in srl_dict["verbs"] if sd['verb'] in verbs]
            srl_dict["verbs"] = s_verbs

        if len(verb_idxs) != len(srl_dict["verbs"]):
            verb_idxs = verb_idxs[1:len(srl_dict["verbs"])+1]

        assert len(verb_idxs) == len(srl_dict["verbs"])

        for vid, verb_entry in enumerate(srl_dict["verbs"]):
            arg_stat = -1
            verb_entry["tags"].append("B")
            for idx, tag in enumerate(verb_entry["tags"]):
                if tag.startswith("B") or tag == "O":
                    if arg_stat != -1:
                        has_verb = False
                        if arg_stat == 0:
                            a = 1
                        if idx - arg_stat > 4:
                            for wd_idx in range(arg_stat, idx):
                                for verb_idx in verb_idxs:
                                    if wd_idx == verb_idx[0]:
                                        has_verb = True
                                        for j in range(verb_idx[0], verb_idx[1]):
                                            edge[verb_idxs[vid][0]][j] = 1
                                            edge[j][verb_idxs[vid][0]] = 1
                                        break
                        #if not has_verb:
                        #    edge[verb_idxs[vid][0]][arg_stat] = 1
#
                        #    edge[arg_stat][verb_idxs[vid][0]] = 1
                        #    for i in range(arg_stat, idx):
                        #        for j in range(i+1, idx):
                        #            edge[i][j] = 1
                        #            edge[j][i] = 1

                        if not has_verb:
                            edge[verb_idxs[vid][0]][arg_stat] = 1

                            edge[arg_stat][verb_idxs[vid][0]] = 1
                            for i in range(arg_stat, idx):
                                for j in range(i+1, idx):
                                    edge[i][j] = 1
                                    edge[j][i] = 1

                    if tag == "B-V" or tag == "O":
                        arg_stat = -1
                    else:
                        arg_stat = idx
                elif tag.startswith("I"):
                    continue
                else:
                    assert False, print(tag)
        #print(edge)
        edges.append(edge[:sent_len, :sent_len])

    word_num = len(token_lst)
    lst_edge = _merge_edge_lst(edges, word_num)
    lst_edge = edge_func(lst_edge, word_num, sent_lens, token_lst, entry, args)

    sparse_edges = sparse.csr_matrix(lst_edge)
    graph = dgl.from_scipy(sparse_edges)
    return graph, graph_words

def get_fc_graph(srl_dicts, words, sent_lens, entry, args):
    word_num = sum(sent_lens)
    lst_edge = np.ones([word_num, word_num])
    sparse_edges = sparse.csr_matrix(lst_edge)
    graph = dgl.from_scipy(sparse_edges)
    return graph, None

def get_simple_srl_graph(srl_dicts, words, sent_lens, entry, args):
    token_lst = []
    base_idx = 0

    for srl_dict in srl_dicts:
        token_lst += srl_dict['words']
    length = len(token_lst)
    edges = np.eye(length)

    for srl_dict in srl_dicts:
        words = srl_dict['words']
        # self-loop
        # link with words in its args
        for verb_dict in srl_dict['verbs']:
            for i, word in enumerate(words):
                if verb_dict['verb'] == word:
                    tags = verb_dict['tags']
                    for j, arg in enumerate(tags):
                        if arg != 'O':
                            edges[base_idx + i][base_idx + j] = edges[base_idx + j][base_idx + i] = 1
        base_idx += len(words)

    word_num = length
    lst_edge = _merge_edge_lst(edges, word_num)
    lst_edge = _add_edges(lst_edge, word_num, sent_lens, token_lst, entry, args)

    sparse_edges = sparse.csr_matrix(edges)
    graph = dgl.from_scipy(sparse_edges)
    # print(edges)
    # print(sparse_edges)
    # print(graph)
    return graph


def get_fully_claim_graph(srl_dicts, words, sent_lens, entry, args):
    word_num = sum(sent_lens)
    lst_edge = np.zeros([word_num, word_num])
    lst_edge[:sent_lens[0], :sent_lens[0]] = np.ones([sent_lens[0], sent_lens[0]])
    sparse_edges = sparse.csr_matrix(lst_edge)
    graph = dgl.from_scipy(sparse_edges)
    return graph, None

def get_gmn_graph(srl_dicts, words, sent_lens, entry, args):
    word_num = sum(sent_lens)
    lst_edge = np.zeros([word_num, word_num])
    lst_edge = _add_self_loop(lst_edge, sent_lens[0]) #?
    lst_edge = _add_gmn(lst_edge, word_num, sent_lens)
    sparse_edges = sparse.csr_matrix(lst_edge)
    graph = dgl.from_scipy(sparse_edges)
    return graph, None

def get_srl_evi_graph(srl_dicts, words, sent_lens, entry, args):
    def _add_edges_srl_evis(lst_edge, word_num, sent_lens, token_lst, entry, args):
        lst_edge = _add_self_loop(lst_edge, word_num)
        new_lst_edge = np.zeros_like(lst_edge, dtype=lst_edge.dtype)
        new_lst_edge[sent_lens[0]:, sent_lens[0]:] = lst_edge[sent_lens[0]:, sent_lens[0]:]
        return new_lst_edge
    return get_srl_graph(srl_dicts, words, sent_lens, entry, args, edge_func = _add_edges_srl_evis)

def _add_gmn(edges, word_num, sent_lens):
    for i in range(sent_lens[0], word_num):
        for j in range(sent_lens[0]):
            edges[i][j] = 1
    return edges

def _add_self_loop(edges, word_num):
    for i in range(word_num):
        edges[i][i] = 1
    return edges

def _add_fully_claim(edges, sent_lens):
    for i in range(sent_lens[0]):
        for j in range(sent_lens[0]):
            edges[i][j] = 1
    return edges

def _add_overlap(edges, token_lst):
    for i in range(len(token_lst)):
        for j in range(i+1, len(token_lst)):
            if overlap(token_lst[i], token_lst[j]):
                edges[i][j] = edges[j][i] = 1
    return edges

def _add_adjacent(edges, token_lst):
    for i in range(len(token_lst) - 1):
        edges[i][i + 1] = 1
        edges[i + 1][i] = 1
    return edges

def _add_fc(edges):
    return np.ones_like(edges, dtype=edges.dtype)

def _add_cluster_words(entry, token_lst, lst_edge):
    # cluster words with same tag
    tag_lst = [[] for _ in range(len(token_lst))]
    upper_lst = []
    for i, wd in enumerate(entry['claim_words']):
        if not wd in entry['claim']:
            upper_lst.append('O')
        else:
            upper_lst.append("UPPER")

    for evi_words, evi_w, evi_t in zip(entry["evidences_words"], entry["evidences"], entry["evidences_title"]):
        evi = evi_t + " : " + evi_w
        for wd in evi_words:
            if not wd in evi:
                upper_lst.append('O')
            else:
                upper_lst.append("UPPER")
    assert len(upper_lst) == len(tag_lst)

    ner_lst = entry['claim_dep']["ner"]
    for evi in entry["evidences_dep"]:
        ner_lst += evi["ner"]
    ner_lst = [tag[2:] if tag.startswith("B") or tag.startswith("S") else tag for tag in ner_lst]
    assert len(ner_lst) == len(tag_lst)

    pos_lst = entry["claim_dep"]["upos"]
    for evi in entry["evidences_dep"]:
        pos_lst += evi["upos"]
    assert len(pos_lst) == len(tag_lst)

    tag_bag = list(zip(upper_lst, ner_lst, pos_lst))
    cluster_lst = ["UPPER",
                   "VERB", "ADJ", "ADV", "NOUN", "PRON",
                   "PERSON", "DATE", "CARDINAL", "ORDINAL", "ORG", "GPE"]
    cluster_dict = {}
    for cls_name in cluster_lst:
        cluster_dict[cls_name] = []
    for i, tags in enumerate(tag_bag):
        for tag in tags:
            if tag in cluster_dict:
                for j in cluster_dict[tag]:
                    lst_edge[i][j] = 1
                    lst_edge[j][i] = 1
                cluster_dict[tag].append(i)
    return lst_edge

def _merge_edge_lst(edge_lst, word_num):
    lst_edge = np.zeros([word_num, word_num])
    # print(lst_edge)
    base_idx = 0
    for edge in edge_lst:
        lst_edge[base_idx:base_idx + edge.shape[0], base_idx:base_idx + edge.shape[0]] = edge
        base_idx += edge.shape[0]
    return lst_edge


def overlap(x, y):
    return x.lower() == y.lower()

if __name__ == "__main__":
    srl_dict = {
        "verbs": [
            {
                "verb": "became",
                "description": "[ARG1: Colin Kaepernick] [V: became] [ARG2: a starting quarterback] [ARGM-TMP: during the 49ers 63rd season in the National Football League] .",
                "tags": [
                    "B-ARG1",
                    "I-ARG1",
                    "B-V",
                    "B-ARG2",
                    "I-ARG2",
                    "I-ARG2",
                    "B-ARGM-TMP",
                    "I-ARGM-TMP",
                    "I-ARGM-TMP",
                    "I-ARGM-TMP",
                    "I-ARGM-TMP",
                    "I-ARGM-TMP",
                    "I-ARGM-TMP",
                    "I-ARGM-TMP",
                    "I-ARGM-TMP",
                    "I-ARGM-TMP",
                    "O"
                ]
            },
            {
                "verb": "starting",
                "description": "Colin Kaepernick became a [V: starting] [ARG1: quarterback] during the 49ers 63rd season in the National Football League .",
                "tags": [
                    "O",
                    "O",
                    "O",
                    "O",
                    "B-V",
                    "B-ARG1",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O"
                ]
            }
        ],
        "words": [
            "Colin",
            "Kaepernick",
            "became",
            "a",
            "starting",
            "quarterback",
            "during",
            "the",
            "49ers",
            "63rd",
            "season",
            "in",
            "the",
            "National",
            "Football",
            "League",
            "."
        ]
    }
    evidence_srl = [
        {
            "verbs": [
                {
                    "verb": "remained",
                    "description": "[ARG1: He] [V: remained] [ARG3: the team 's starting quarterback] [ARGM-TMP: for the rest of the season] and went on to lead the 49ers to their first Super Bowl appearance since 1994 , losing to the Baltimore Ravens . quarterback quarterback Super Bowl Super Bowl XLVII 1994 Super Bowl XXIX Baltimore Ravens Baltimore Ravens",
                    "tags": [
                        "B-ARG1",
                        "B-V",
                        "B-ARG3",
                        "I-ARG3",
                        "I-ARG3",
                        "I-ARG3",
                        "I-ARG3",
                        "B-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O"
                    ]
                },
                {
                    "verb": "starting",
                    "description": "He remained the team 's [V: starting] [ARG1: quarterback] [ARGM-TMP: for the rest of the season] and went on to lead the 49ers to their first Super Bowl appearance since 1994 , losing to the Baltimore Ravens . quarterback quarterback Super Bowl Super Bowl XLVII 1994 Super Bowl XXIX Baltimore Ravens Baltimore Ravens",
                    "tags": [
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "B-V",
                        "B-ARG1",
                        "B-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O"
                    ]
                },
                {
                    "verb": "went",
                    "description": "[ARG0: He] remained the team 's starting quarterback for the rest of the season and [V: went] on [ARGM-PRP: to lead the 49ers to their first Super Bowl appearance since 1994] , [ARGM-PRD: losing to the Baltimore Ravens . quarterback quarterback Super Bowl Super Bowl XLVII 1994 Super Bowl XXIX Baltimore Ravens Baltimore Ravens]",
                    "tags": [
                        "B-ARG0",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "B-V",
                        "O",
                        "B-ARGM-PRP",
                        "I-ARGM-PRP",
                        "I-ARGM-PRP",
                        "I-ARGM-PRP",
                        "I-ARGM-PRP",
                        "I-ARGM-PRP",
                        "I-ARGM-PRP",
                        "I-ARGM-PRP",
                        "I-ARGM-PRP",
                        "I-ARGM-PRP",
                        "I-ARGM-PRP",
                        "I-ARGM-PRP",
                        "O",
                        "B-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD"
                    ]
                },
                {
                    "verb": "lead",
                    "description": "[ARG0: He] remained the team 's starting quarterback for the rest of the season and went on to [V: lead] [ARG1: the 49ers] [ARG4: to their first] [ARG2: Super Bowl appearance] [ARGM-TMP: since 1994] , losing to the Baltimore Ravens . quarterback quarterback Super Bowl Super Bowl XLVII 1994 Super Bowl XXIX Baltimore Ravens Baltimore Ravens",
                    "tags": [
                        "B-ARG0",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "B-V",
                        "B-ARG1",
                        "I-ARG1",
                        "B-ARG4",
                        "I-ARG4",
                        "I-ARG4",
                        "B-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "B-ARGM-TMP",
                        "I-ARGM-TMP",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O"
                    ]
                },
                {
                    "verb": "losing",
                    "description": "[ARG0: He] remained the team 's starting quarterback for the rest of the season and went on to lead the 49ers to their first Super Bowl appearance since 1994 , [V: losing] [ARG2: to the Baltimore Ravens . quarterback quarterback Super Bowl Super Bowl XLVII 1994 Super Bowl XXIX Baltimore Ravens Baltimore Ravens]",
                    "tags": [
                        "B-ARG0",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "B-V",
                        "B-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "I-ARG2"
                    ]
                }
            ],
            "words": [
                "He",
                "remained",
                "the",
                "team",
                "'s",
                "starting",
                "quarterback",
                "for",
                "the",
                "rest",
                "of",
                "the",
                "season",
                "and",
                "went",
                "on",
                "to",
                "lead",
                "the",
                "49ers",
                "to",
                "their",
                "first",
                "Super",
                "Bowl",
                "appearance",
                "since",
                "1994",
                ",",
                "losing",
                "to",
                "the",
                "Baltimore",
                "Ravens",
                ".",
                "quarterback",
                "quarterback",
                "Super",
                "Bowl",
                "Super",
                "Bowl",
                "XLVII",
                "1994",
                "Super",
                "Bowl",
                "XXIX",
                "Baltimore",
                "Ravens",
                "Baltimore",
                "Ravens"
            ]
        },
        {
            "verbs": [
                {
                    "verb": "began",
                    "description": "[ARG0: Kaepernick] [V: began] [ARG1: his professional career] [ARGM-PRD: as a backup to Alex Smith] , but became the 49ers ' starter in the middle of the 2012 season after Smith suffered a concussion . Alex Smith Alex Smith 2012 season 2012 San Francisco 49ers season concussion concussion",
                    "tags": [
                        "B-ARG0",
                        "B-V",
                        "B-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "B-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "I-ARGM-PRD",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O"
                    ]
                },
                {
                    "verb": "became",
                    "description": "[ARG1: Kaepernick] began his professional career as a backup to Alex Smith , but [V: became] [ARG2: the 49ers ' starter] [ARGM-TMP: in the middle of the 2012 season] [ARGM-TMP: after Smith suffered a concussion . Alex Smith Alex Smith 2012 season 2012 San Francisco 49ers season concussion concussion]",
                    "tags": [
                        "B-ARG1",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "B-V",
                        "B-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "B-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "B-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP"
                    ]
                },
                {
                    "verb": "suffered",
                    "description": "Kaepernick began his professional career as a backup to Alex Smith , but became the 49ers ' starter in the middle of the 2012 season after [ARG0: Smith] [V: suffered] [ARG1: a concussion] . Alex Smith Alex Smith 2012 season 2012 San Francisco 49ers season concussion concussion",
                    "tags": [
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "B-ARG0",
                        "B-V",
                        "B-ARG1",
                        "I-ARG1",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O"
                    ]
                }
            ],
            "words": [
                "Kaepernick",
                "began",
                "his",
                "professional",
                "career",
                "as",
                "a",
                "backup",
                "to",
                "Alex",
                "Smith",
                ",",
                "but",
                "became",
                "the",
                "49ers",
                "'",
                "starter",
                "in",
                "the",
                "middle",
                "of",
                "the",
                "2012",
                "season",
                "after",
                "Smith",
                "suffered",
                "a",
                "concussion",
                ".",
                "Alex",
                "Smith",
                "Alex",
                "Smith",
                "2012",
                "season",
                "2012",
                "San",
                "Francisco",
                "49ers",
                "season",
                "concussion",
                "concussion"
            ]
        },
        {
            "verbs": [
                {
                    "verb": "helped",
                    "description": "[ARGM-TMP: During the 2013 season , his first full season as a starter ,] [ARG0: Kaepernick] [V: helped] [ARG1: the 49ers reach the NFC Championship , losing to the Seattle Seahawks . 2013 season 2013 San Francisco 49ers season NFC Championship 2013–14 NFL playoffs#NFC Championship Game] : Seattle Seahawks 23.2C San Francisco 49ers 17 Seattle Seahawks Seattle Seahawks",
                    "tags": [
                        "B-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "B-ARG0",
                        "B-V",
                        "B-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O"
                    ]
                },
                {
                    "verb": "reach",
                    "description": "During the 2013 season , his first full season as a starter , Kaepernick helped [ARG0: the 49ers] [V: reach] [ARG1: the NFC Championship] , [ARGM-ADV: losing to the Seattle Seahawks . 2013 season 2013 San Francisco 49ers season NFC Championship 2013–14 NFL playoffs#NFC Championship Game] : Seattle Seahawks 23.2C San Francisco 49ers 17 Seattle Seahawks Seattle Seahawks",
                    "tags": [
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "B-ARG0",
                        "I-ARG0",
                        "B-V",
                        "B-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "O",
                        "B-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O"
                    ]
                },
                {
                    "verb": "losing",
                    "description": "During the 2013 season , his first full season as a starter , [ARG0: Kaepernick] helped the 49ers reach the NFC Championship , [V: losing] [ARG2: to the Seattle Seahawks] . 2013 season 2013 San Francisco 49ers season NFC Championship 2013–14 NFL playoffs#NFC Championship Game : Seattle Seahawks 23.2C San Francisco 49ers 17 Seattle Seahawks Seattle Seahawks",
                    "tags": [
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "B-ARG0",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "B-V",
                        "B-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "I-ARG2",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O"
                    ]
                }
            ],
            "words": [
                "During",
                "the",
                "2013",
                "season",
                ",",
                "his",
                "first",
                "full",
                "season",
                "as",
                "a",
                "starter",
                ",",
                "Kaepernick",
                "helped",
                "the",
                "49ers",
                "reach",
                "the",
                "NFC",
                "Championship",
                ",",
                "losing",
                "to",
                "the",
                "Seattle",
                "Seahawks",
                ".",
                "2013",
                "season",
                "2013",
                "San",
                "Francisco",
                "49ers",
                "season",
                "NFC",
                "Championship",
                "2013–14",
                "NFL",
                "playoffs#NFC",
                "Championship",
                "Game",
                ":",
                "Seattle",
                "Seahawks",
                "23.2C",
                "San",
                "Francisco",
                "49ers",
                "17",
                "Seattle",
                "Seahawks",
                "Seattle",
                "Seahawks"
            ]
        },
        {
            "verbs": [
                {
                    "verb": "following",
                    "description": "In the [V: following] [ARG1: seasons] , Kaepernick lost and won back his starting job , with the 49ers missing the playoffs for three years consecutively .",
                    "tags": [
                        "O",
                        "O",
                        "B-V",
                        "B-ARG1",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O"
                    ]
                },
                {
                    "verb": "lost",
                    "description": "[ARGM-TMP: In the following seasons] , [ARG0: Kaepernick] [V: lost] and won back [ARG1: his starting job] , [ARGM-ADV: with the 49ers missing the playoffs for three years consecutively] .",
                    "tags": [
                        "B-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "O",
                        "B-ARG0",
                        "B-V",
                        "O",
                        "O",
                        "O",
                        "B-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "O",
                        "B-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "O"
                    ]
                },
                {
                    "verb": "won",
                    "description": "[ARGM-TMP: In the following seasons] , [ARG0: Kaepernick] lost and [V: won] [ARGM-DIR: back] [ARG1: his starting job] , [ARGM-ADV: with the 49ers missing the playoffs for three years consecutively] .",
                    "tags": [
                        "B-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "O",
                        "B-ARG0",
                        "O",
                        "O",
                        "B-V",
                        "B-ARGM-DIR",
                        "B-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "O",
                        "B-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "I-ARGM-ADV",
                        "O"
                    ]
                },
                {
                    "verb": "starting",
                    "description": "In the following seasons , Kaepernick lost and won back his [V: starting] [ARG1: job] , with the 49ers missing the playoffs for three years consecutively .",
                    "tags": [
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "B-V",
                        "B-ARG1",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O"
                    ]
                },
                {
                    "verb": "missing",
                    "description": "In the following seasons , Kaepernick lost and won back his starting job , with [ARG0: the 49ers] [V: missing] [ARG1: the playoffs] [ARGM-TMP: for three years consecutively] .",
                    "tags": [
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "B-ARG0",
                        "I-ARG0",
                        "B-V",
                        "B-ARG1",
                        "I-ARG1",
                        "B-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "O"
                    ]
                }
            ],
            "words": [
                "In",
                "the",
                "following",
                "seasons",
                ",",
                "Kaepernick",
                "lost",
                "and",
                "won",
                "back",
                "his",
                "starting",
                "job",
                ",",
                "with",
                "the",
                "49ers",
                "missing",
                "the",
                "playoffs",
                "for",
                "three",
                "years",
                "consecutively",
                "."
            ]
        },
        {
            "verbs": [
                {
                    "verb": "born",
                    "description": "[ARG1: Colin Rand Kaepernick] -LRB- -LSB- ` kæpərnɪk -RSB- ; [V: born] [ARGM-TMP: November 3 , 1987] -RRB- is an American football quarterback who is currently a free agent . American football American football quarterback quarterback",
                    "tags": [
                        "B-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "B-V",
                        "B-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O"
                    ]
                }
            ],
            "words": [
                "Colin",
                "Rand",
                "Kaepernick",
                "-LRB-",
                "-LSB-",
                "`",
                "kæpərnɪk",
                "-RSB-",
                ";",
                "born",
                "November",
                "3",
                ",",
                "1987",
                "-RRB-",
                "is",
                "an",
                "American",
                "football",
                "quarterback",
                "who",
                "is",
                "currently",
                "a",
                "free",
                "agent",
                ".",
                "American",
                "football",
                "American",
                "football",
                "quarterback",
                "quarterback"
            ]
        }
    ]
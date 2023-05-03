# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/9/16 14:47
# Description:
import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_trf")

def split_conj(sent):
    sel_relations = ["conj"]
    def dye_false(token):
        if token_dict[token.head][2] == False and (token.dep_ not in sel_relations):
            token_dict[token][2] = False
        for child in token.children:
            dye_false(child)
    '''
    def find_conj_sets(token, conj_sets):
        has_conj_child = 0
        for child in token.children:
            if child.dep_ in sel_relations:
                has_conj_child += 1
            find_conj_sets(child, conj_sets)
        assert has_conj_child <= 1

        if has_conj_child == 0 and token.dep_ in sel_relations:
            conj_set = []
            tmp_token = token
            while tmp_token.dep_ in sel_relations:
                conj_set.append(tmp_token)
                tmp_token = tmp_token.head
            conj_set.append(tmp_token)
            conj_sets.append(conj_set.copy())
        return
    '''

    doc = nlp(sent)
    #displacy.render(doc, style="dep")

    #for token in doc:
    #    print(token.text, token.dep_, token.head.text, token.head.pos_,
    #            [child for child in token.children])

    token_dict = {}
    for idx, token in enumerate(doc):
        token_dict[token] = [idx, token, True]

    conj_sets = []
    for token in doc:
        if token.dep_ != "conj" and token.conjuncts:
            conj_set = list(token.conjuncts)
            conj_set.insert(0, token)
            conj_sets.append(conj_set)

    roots = []
    for token in doc:
        if token.dep_ == "ROOT":
            #find_conj_sets(token, conj_sets)
            roots.append(token)
            #break

    #conj_sets[0]
    conj_sents = []
    for conj_set in conj_sets:
        if len(conj_set) > 3:
            continue
        for op in conj_set:
            sent_lst = []
            for k in token_dict:
                token_dict[k][2] = True
            for option in conj_set:
                if option == op:
                    token_dict[option][2] = True
                else:
                    token_dict[option][2] = False
            for root in roots:
                dye_false(root)
            for child in op.children:
                if child.text == "," or child.text == "and":
                    token_dict[child][2] = False
            for k,v in token_dict.items():
                if v[2]:
                    sent_lst.append([v[0], v[1].text])
            conj_sents.append(sent_lst)

    sents = []
    #for sent_lst in conj_sents[::-1]:
    for sent_lst in conj_sents:
        sent_lst.sort(key=lambda x: x[0])
        sent = ' '.join([obj[1] for obj in sent_lst])
        #print(sent)
        sents.append(sent)

    if not sents:
        sents = [sent]
    return sents, doc

if __name__ == "__main__":
    sent = "Kentucky is also known for horse racing , bourbon distilleries , coal , the historic site My Old Kentucky Home , automobile manufacturing , tobacco , bluegrass music , college basketball , and Kentucky Fried Chicken ."
    while True:
        sents, _ = split_conj(sent)
        for s in sents:
            print(s)
        sent = input()

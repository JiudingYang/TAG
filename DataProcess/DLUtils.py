import networkx as nx
import torch
import ujson
import random


def loadRaw(path):
    # load row data for this project
    Data = []
    Error = []
    with open(path, 'r', encoding='utf-8') as file:
        for lines in file:
            try:
                Data.append(ujson.loads(lines))
            except:
                Error.append(lines)
    with open('Data/error.txt', 'w', encoding='utf-8') as file:
        for lines in Error:
            file.write(lines)
    return Data


def get_word_pos_dict(sentence, concept, nlp_model):
    # return list of list of dicts
    return nlp_model(sentence).to_dict(), nlp_model(concept).to_dict(),


def build_semanticGraph(sentence_pos_dict, concept_pos_dict):
    # input list of list of dicts
    semanticGraph = nx.DiGraph()
    # first add all words as nodes
    for list_item in sentence_pos_dict:
        for word in list_item:
            semanticGraph.add_node(word['text'], word=word['text'], upos=word['upos'])

    for list_item in concept_pos_dict:
        for word in list_item:
            semanticGraph.add_node(word['text'], word=word['text'], upos=word['upos'])
    # them add edges

    for list_item in concept_pos_dict:
        for word in list_item:
            relation = word['deprel']
            if relation == 'root':
                continue
            head = list_item[word['head'] - 1]['text']
            if (head, word['text']) in semanticGraph.edges():
                continue
            semanticGraph.add_edge(head, word['text'], rel=relation)
            semanticGraph.add_edge(word['text'], head, rel='-' + relation)

    for list_item in sentence_pos_dict:
        for word in list_item:
            relation = word['deprel']
            if relation == 'root':
                continue
            head = list_item[word['head'] - 1]['text']
            semanticGraph.add_edge(head, word['text'], rel=relation)
            semanticGraph.add_edge(word['text'], head, rel='-' + relation)

    return semanticGraph


def get_graph(sentence, concept, nlp_model):
    sentence_pos_dict, concept_pos_dict = get_word_pos_dict(sentence, concept, nlp_model)
    graph = build_semanticGraph(sentence_pos_dict, concept_pos_dict)
    return graph


def Bert_tokenizer(sentence, tokenizer):
    return torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)])


def gen_unique_set(data, ratio):
    # generate two set of data that has different concepts
    concept_list = {}
    for row in data:
        concept_list[row['concept']] = {}
    for row in data:
        concept_list[row['concept']][row['sentence']] = row

    # fist balancing the concept that has multiple sentences
    mul_concept = {}
    for item in concept_list.keys():
        if len(concept_list[item].keys()) > 1:
            mul_concept[item] = concept_list[item]

    list = []
    for item in mul_concept.keys():
        list.append(item)
    random.shuffle(list)

    L = len(list) * 0.875
    L = int(L)
    train_list = list[:L]
    valid_list = list[L:]
    train_data = []
    valid_data = []
    for item in valid_list:
        for sentence in mul_concept[item].keys():
            valid_data.append(mul_concept[item][sentence])
    for item in train_list:
        for sentence in mul_concept[item].keys():
            train_data.append(mul_concept[item][sentence])

    # then add concepts that has single sentence
    sin_concept = {}
    for item in concept_list.keys():
        if len(concept_list[item].keys()) == 1:
            sin_concept[item] = concept_list[item]
    list = []
    for item in sin_concept.keys():
        list.append(item)
    random.shuffle(list)
    L = len(data)*ratio - len(valid_data)
    L = int(L)
    valid_list = list[:L]
    train_list = list[L:]

    for item in valid_list:
        for sentence in sin_concept[item].keys():
            valid_data.append(sin_concept[item][sentence])

    for item in train_list:
        for sentence in sin_concept[item].keys():
            train_data.append(sin_concept[item][sentence])

    return train_data, valid_data

# if __name__ == '__main__':
#     from pylab import *
#     import pandas as pd
#     edge_labels = {}
#     node_labels = {}
#
#     mpl.rcParams['font.sans-serif'] = ['SimSun']
#
#     graph = get_graph('今天下雨，我不想去上课')
#
#     pos = nx.kamada_kawai_layout(graph)
#
#     for node in graph.nodes():
#         node_labels[node] = node
#     nx.draw(graph, pos, labels=node_labels)
#     nx.draw_networkx_edge_labels(graph, pos, labels=nx.get_edge_attributes(graph, 'rel'))
#
#     plt.show()

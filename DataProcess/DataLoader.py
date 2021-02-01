import os
import stanza
import _pickle as pk
import transformers as trans
from configs import *
from numpy import mean
from DataProcess.DLUtils import *
from torch_geometric.utils import from_networkx
from UtilPack.DataUtil import cut_dict, gen_tag_emb_from_dict, load_embs


def get_word_emb(data=None):
    # input is list of dict, imported from json
    # ---------------------Word--------------------------
    # whether cut DICT size
    if not DataConfig.IS_DICT_CUT:
        if data is None:
            print('Dict is not cut, please input data to generate new word dict')
        print("Cutting Glove Dict")
        all_words = {}
        # collecting all words in raw data
        for row in data:
            for word_list in row['others']['concept_segs']:
                all_words[word_list[0]] = 1
            for word_list in row['others']['sentence_segs']:
                all_words[word_list[0]] = 1
            for entity in row['context']['entity']:
                all_words[entity] = 1
            for entity in row['entity'].keys():
                all_words[entity] = 1
            for con in row['context']['concept']:
                for word_list in con[1]:
                    all_words[word_list[0]] = 1

        word_dict, word_mats = cut_dict(all_words.keys(),
                                        DataConfig.NODE_EMBDICT_PATH,
                                        EmbConfig.emb_config['word']['emb_dim'],
                                        save_path=DataConfig.EMB_WORD_PATH)
    else:
        word_config = {'word': EmbConfig.emb_config['word']}
        emb_dict, emb_mats = load_embs(word_config)
        word_dict = emb_dict['word']
        word_mats = emb_mats['word']
        del word_config, emb_dict, emb_mats
    print("Glove Dict Loaded")
    return word_dict, word_mats


def get_tag_emb():
    tag_config = EmbConfig.emb_config.copy()
    del tag_config['word']
    if not DataConfig.IS_TAGEMB_GENERATED:
        # generate embedding dict for different tags of nodes
        print("Generating Tag Embedding Dict")
        emb_dict, _ = gen_tag_emb_from_dict(tag_config)
    else:
        emb_dict, _ = load_embs(tag_config)
    pos_idx_dict = emb_dict['pos']
    source_idx_dict = emb_dict['source']
    ner_idx_dict = emb_dict['ner']
    del tag_config, emb_dict
    print("Tag Embedding Dict Loaded")

    return pos_idx_dict, source_idx_dict, ner_idx_dict


def avg_emb(text, word_mats, word_dict):
    # get average of emb for given text
    word_vec = []
    length = len(word_mats[0])
    word_vec.append([0] * length)
    for word in text:
        if word in word_dict.keys():
            word_vec.append(word_mats[word_dict[word]])
    return mean(word_vec, axis=0)


def add_tags(graph, concept_list, sentence_list, ner_dict, pos_idx_dict,
             source_idx_dict, ner_idx_dict, word_dict, word_mats):
    for node in graph.nodes():
        # initial tag
        source = 'CON'
        ner = 'None'
        # check source tag
        for sen in sentence_list:
            if node in sen:
                source = 'SEN'
                for con in concept_list:
                    if node in con:
                        source = 'BOTH'
                        break
                if source == 'BOTH':
                    break

        # check NER tag
        if node in ner_dict.keys():
            ner = ner_dict[node]
        # check POS tag
        pos = graph.nodes[node]['upos']
        # add tags as attributes of node
        graph.nodes[node]['pos_tag'] = pos_idx_dict[pos]
        graph.nodes[node]['source_tag'] = source_idx_dict[source]
        graph.nodes[node]['ner_tag'] = ner_idx_dict[ner]
        # add word vector emb as 'x'
        if node in word_dict.keys():
            graph.nodes[node]['x'] = word_mats[word_dict[node]]
        else:
            graph.nodes[node]['x'] = [0] * EmbConfig.emb_config['word']['emb_dim']
        # delete pos ans xpos attributes of node
        del graph.nodes[node]['upos']
    return graph


def entity_node_fix(graph, row, sentence_list, ner_dict, pos_idx_dict,
                    source_idx_dict, ner_idx_dict, word_dict, word_mats):
    for ner in ner_dict.keys():
        if ner not in sentence_list[0]:
            # if ner is not in sentence_list, which mean it's parsed
            # find embedding
            _temp_emb = [0] * EmbConfig.emb_len_word
            if ner in word_dict.keys():
                _temp_emb = word_mats[word_dict[ner]]
            # add ner as node
            source_tag = 'SEN'
            if ner in row['entity'].keys():
                source_tag = 'CON'
                for e in row['others']['sentence_ner']:
                    if ner == e['name']:
                        source_tag = 'BOTH'
            graph.add_node(ner,
                           pos_tag=pos_idx_dict['NOUN'],
                           source_tag=source_idx_dict[source_tag],
                           ner_tag=ner_idx_dict[ner_dict[ner]],
                           x=_temp_emb
                           )
            # try to find it's parts
            window_size = len(ner)
            for _index, word in enumerate(sentence_list[0]):
                word_size = len(word)
                index_list = [_index]
                hit = False
                search_index = _index
                while 1:
                    if word_size > window_size:
                        break
                    search_index = search_index - 1
                    index_list.insert(0, search_index)
                    if search_index < 0:
                        break
                    word_size += len(sentence_list[0][search_index])
                    if word_size == window_size:
                        hit = True
                        break
                if hit:
                    # length matched
                    s = ''
                    for iter in index_list:
                        s += sentence_list[0][iter]
                    if s == ner:
                        # ner parts detected, add ner links
                        for iter in index_list:
                            target_node = sentence_list[0][iter]
                            graph.add_edge(ner, target_node, rel='contains')
                            graph.add_edge(target_node, ner, rel='-contains')
    return graph


def get_pyg_data(graph):
    graph_data = from_networkx(graph)
    graph_data.x = graph_data.x.float()
    graph_data.rel = graph_data.rel.long()
    graph_data.edge_index = graph_data.edge_index.long()
    return graph_data


def add_virtual_node(graph, row, concept_list, sentence_list, ner_dict, pos_idx_dict,
                     source_idx_dict, ner_idx_dict, word_mats, word_dict):
    # --------------------------ADD NODES---------------------
    # add subject nodes
    for sub in row['context']['subject']:
        if sub in word_dict.keys():
            emb = word_mats[word_dict[sub]]
        else:
            emb = [0] * EmbConfig.emb_len_word

        graph.add_node('娱乐_' + sub,
                       word='娱乐_' + sub,
                       pos_tag=pos_idx_dict['CONTEXT_SUBJECT'],
                       source_tag=source_idx_dict['CON'],
                       ner_tag=ner_idx_dict['None'],
                       x=emb
                       )

    # add context concepts
    for con in row['context']['concept']:
        graph.add_node(con[0],
                       word=con[0],
                       pos_tag=pos_idx_dict['CONTEXT_CONCEPT'],
                       source_tag=source_idx_dict['CON'],
                       ner_tag=ner_idx_dict['None'],
                       x=avg_emb([word_list[0] for word_list in con[1]], word_mats, word_dict)
                       )
    # add concept virtual node
    concept_emb = avg_emb(concept_list[0], word_mats, word_dict)
    graph.add_node('CONCEPT',
                   word='CONCEPT',
                   pos_tag=pos_idx_dict['CONCEPT'],
                   source_tag=source_idx_dict['CON'],
                   ner_tag=ner_idx_dict['None'],
                   x=concept_emb
                   )
    # add sentence as node, and it's links to all ners from sentence
    sentence_emb = avg_emb(sentence_list[0], word_mats, word_dict)
    graph.add_node('SENTENCE',
                   word='SENTENCE',
                   pos_tag=pos_idx_dict['SENTENCE'],
                   source_tag=source_idx_dict['SEN'],
                   ner_tag=ner_idx_dict['None'],
                   x=sentence_emb
                   )

    # -----------------------ADD EDGES---------------------

    # add links of CONCEPT node
    for word in concept_list[0]:
        graph.add_edge('CONCEPT', word, rel='contains')
        graph.add_edge(word, 'CONCEPT', rel='-contains')

    # add link between context concept and it's words
    for con in row['context']['concept']:
        for word in con[1]:
            graph.add_edge(con[0], word[0], rel='contains')
            graph.add_edge(word[0], con[0], rel='-contains')

    # add links according to context relation
    for relation in row['context']['relation']:
        if relation[0] in graph.nodes() and relation[1] in graph.nodes():
            graph.add_edge(relation[0], relation[1], rel='contains')
            graph.add_edge(relation[1], relation[0], rel='-contains')

    # add entity <--> CONCEPT links
    for e in row['entity'].keys():
        graph.add_edge(e, 'CONCEPT', rel='obtain')
        graph.add_edge('CONCEPT', e, rel='-obtain')

    # add subject <--> concepts
    for sub in row['context']['subject']:
        graph.add_edge('娱乐_'+sub, 'CONCEPT', rel='contains')
        graph.add_edge('CONCEPT', '娱乐_'+sub, rel='-contains')

    # add ner links
    for ner in ner_dict.keys():
        graph.add_edge(ner, 'SENTENCE', rel='talked_in')
        graph.add_edge('SENTENCE', ner, rel='-talked_in')

    return graph


def indexing_edge_relation(graph):
    for edge in graph.edges:
        rel = graph.edges[edge]['rel']
        idx = rel.find(':')
        if idx != -1:
            rel = rel[:idx]
        graph.edges[edge]['rel'] = EmbConfig.rel_dict[rel]
    return graph


def build_preprocess_structure(data, nlp_model):
    print("Data Pre-Processing")
    preprocessed_data = []
    # ----------------------Word--------------------------
    # get word emb info (glove)
    word_dict, word_mats = get_word_emb(data)

    # ----------------------Tag--------------------------
    # load tag index dict to encode each slice of data's graph nodes
    pos_idx_dict, source_idx_dict, ner_idx_dict = get_tag_emb()

    # ------------------Structure------------------------
    # build structure
    total_data = len(data)
    for __index, row in enumerate(data):
        print("\r   Processing  %8s / %-13s" % (__index, total_data), end='')
        # get word lists of concept and sentence, prepared for graph building
        concept_list = [[word_list[0] for word_list in row['others']['concept_segs']]]
        # add context concepts
        for con in row['context']['concept']:
            concept_list.append([word_list[0] for word_list in con[1]])

        sentence_list = [[word_list[0] for word_list in row['others']['sentence_segs']]]

        # get ner dict, helps to check ner type for each node in graph
        ner_dict = {}
        for ner in row['others']['sentence_ner']:
            ner_dict[ner['name']] = str(ner['etype'])[:2]
        for ner in row['entity'].keys():
            ner_dict[ner] = str(row['entity'][ner])[:2]

        # ------------------Graph------------------------
        # start adding tags to nodes
        graph = get_graph(sentence_list, concept_list, nlp_model)

        graph = add_tags(graph, concept_list, sentence_list, ner_dict, pos_idx_dict,
                         source_idx_dict, ner_idx_dict, word_dict, word_mats)

        # perform Ner nn fix
        graph = entity_node_fix(graph, row, sentence_list, ner_dict, pos_idx_dict,
                                source_idx_dict, ner_idx_dict, word_dict, word_mats)

        # add concept as node, and it's links to all nodes from concept
        graph = add_virtual_node(graph, row, concept_list, sentence_list, ner_dict, pos_idx_dict,
                                 source_idx_dict, ner_idx_dict, word_mats, word_dict)

        # indexing relation of edges
        graph = indexing_edge_relation(graph)

        # ------------------PyG Data---------------------
        graph_data = get_pyg_data(graph)

        # ------------------Build------------------------
        # gather information and store
        item = {'label': row['label'],
                'concept': row['concept'],
                'sentence': row['sentence'],
                'graph_data': graph_data,
                }

        preprocessed_data.append(item)

    print('  -----Complete!')
    return preprocessed_data


def pre_process_data(_path, _save_path, device):
    if device != 'cpu':
        index = device.index
        torch.cuda.set_device(index)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)

    nlp = stanza.Pipeline(lang='zh-hans', tokenize_pretokenized=True)
    # model = trans.BertModel.from_pretrained('bert-base-chinese')

    _data = build_preprocess_structure(loadRaw(_path), nlp)
    with open(_save_path, 'wb') as f:
        pk.dump(_data, f)
    return _data

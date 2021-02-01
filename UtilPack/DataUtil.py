# ========================================
# utilities for data processing
# ========================================
import _pickle as pk
import numpy as np

def load_dict(path):
    emb_idx_dict = {}
    emb_mat = []
    index = 0
    with open(path, 'r', encoding='utf-8') as file:
        for lines in file:
            row = lines.split()
            emb_idx_dict[row[0]] = index
            emb_mat.append([float(i) for i in row[1:]])
            index += 1
    return emb_mat, emb_idx_dict


def cut_dict(word_list, dict_path, emb_vec_len, save_path=None):
    """
    dict must be a txt-like file that in each row:
    ['Word v1 v2 .... vn']
    where n is the size of embedding vector of each word
    :param word_list: list of all words that contained in raw data
    :param dict_path: glove dict file path
    :param emb_vec_len: length of embedding vector for each word
    :param save_path: if not None, the result will be saved through the path
    :return embedding index dict, embedding vectors in list of list
    """
    with open(dict_path, 'r', encoding='UTF-8') as f:
        _emb_mats = []
        _emb_idx_dict = {}
        index = 0
        while 1:
            line = f.readline()
            if line == '':
                break
            line = line.split()
            # fix case where a word contains ' '
            word = ' '.join(line[0:len(line) - emb_vec_len])
            if word in word_list:
                vector = line[len(line) - emb_vec_len:]
                _emb_mats.append([float(i) for i in vector])
                _emb_idx_dict[word] = index
                index += 1
    if save_path is not None:
        with open(save_path, 'wb') as file:
            pk.dump(_emb_idx_dict, file)
            pk.dump(_emb_mats, file)

    return _emb_idx_dict, _emb_mats


def gen_tag_emb_from_dict(emb_config):
    _emb_dict = {}
    _emb_mats = {}
    for tag in emb_config.keys():
        _dict, _mats = gen_tag_emb(emb_config[tag]['tag_list'], emb_config[tag]['emb_dim'], emb_config[tag]['emb_file_path'])
        _emb_dict[tag] = _dict
        _emb_mats[tag] = _mats
    return _emb_dict, _emb_mats


def gen_tag_emb(tag_list, emb_vec_len, save_path=None):
    _emb_dict = {}
    _emb_mats = []
    for index, tag in enumerate(tag_list):
        _emb_dict[tag] = index
        _emb_mats.append([np.random.normal(scale=0.1) for _ in range(emb_vec_len)])

    # check whether to save generated mats and dict
    if save_path is not None:
        with open(save_path, 'wb') as file:
            pk.dump(_emb_dict, file)
            pk.dump(_emb_mats, file)

    return _emb_dict, _emb_mats


def load_embs(emb_config):
    """
    load all kinds of embedded tags' mats and dict
    :param emb_config: see configs.EmbConfig.emb_config
    :return: emb_dict,emb_mats in format: dict - {'tag_set1':tag_set1_dict, ...}, mats - {'tag_set1':tag_set1_mats, ...}
    """
    _emb_dict = {}
    _emb_mats = {}
    for tag in emb_config.keys():
        path = emb_config[tag]['emb_file_path']
        with open(path, 'rb') as f:
            _dict = pk.load(f)
            _mats = pk.load(f)
        _emb_dict[tag] = _dict
        _emb_mats[tag] = _mats
    return _emb_dict, _emb_mats

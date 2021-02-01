# ------------------------------------
# configs used in project
# ------------------------------------
import torch

UNIVERSAL_SEED = 999
# devise
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'
class DataConfig:
    # Bools
    IS_TAGEMB_GENERATED = True
    IS_PREPROCESSED = False
    IS_DICT_CUT = True
    USE_RAND_DATA = False
    # File Names
    BEST_PARA = 'Graph_Para.mod'
    if not USE_RAND_DATA:
        PREPRO_TRAIN_DATA = 'prepro_train_set.dat'
        PREPRO_TEST_DATA = 'prepro_test_set.dat'
        TRAIN_DATA = 'train_set.json'
        TEST_DATA = 'test_set.json'
    else:
        PREPRO_TRAIN_DATA = 'prepro_train_set_rand.dat'
        PREPRO_TEST_DATA = 'prepro_test_set_rand.dat'
        TRAIN_DATA = 'train_set_rand.json'
        TEST_DATA = 'test_set_rand.json'

    # Node Emb File Names
    NODE_EMBDICT_FILENAME = 'Tencent_AILab_ChineseEmbedding'
    NODE_EMBDICT_PROCESSED_FILENAME = 'word_dict.dat'

    EMB_POS_FILE = 'emb_pos.dat'
    EMB_SOURCE_FILE = 'emb_source.dat'
    EMB_NER_FILE = 'emb_ner.dat'

    # Paths
    DATA_PATH = 'Data/'
    RAW_DATA_FOLDER = DATA_PATH + 'Raw_Data/'
    EMB_DICT_FOLDER = DATA_PATH + 'Embedding_Dict/'
    PROCESSED_DATA_FOLDER = DATA_PATH + 'Processed/'
    MODEL_PARA_FOLDER = DATA_PATH + 'Model_Para/'

    # Generate File Path
    PREPRO_TRAIN_PATH = PROCESSED_DATA_FOLDER + PREPRO_TRAIN_DATA
    PREPRO_TEST_PATH = PROCESSED_DATA_FOLDER + PREPRO_TEST_DATA
    TRAIN_DATA_PATH = RAW_DATA_FOLDER + TRAIN_DATA
    TEST_DATA_PATH = RAW_DATA_FOLDER + TEST_DATA

    # Node Emb Save Path
    NODE_EMBDICT_PATH = EMB_DICT_FOLDER + NODE_EMBDICT_FILENAME

    EMB_POS_PATH = EMB_DICT_FOLDER + EMB_POS_FILE
    EMB_SOURCE_PATH = EMB_DICT_FOLDER + EMB_SOURCE_FILE
    EMB_NER_PATH = EMB_DICT_FOLDER + EMB_NER_FILE
    EMB_WORD_PATH = EMB_DICT_FOLDER + NODE_EMBDICT_PROCESSED_FILENAME

    # Best model parameters
    BEST_PARA_PATH = MODEL_PARA_FOLDER + BEST_PARA

# =========================================================================
# Embedding Settings
# =========================================================================


class EmbConfig:
    # Node Embedding Length
    emb_len_word = 200
    emb_len_pos = 16
    emb_len_source = 8
    emb_len_ner = 16
    emb_tag_len_total = emb_len_pos + emb_len_source + emb_len_ner
    emb_len_total = emb_tag_len_total + emb_len_word

    # Node POS list
    pos_list = [
        'CONCEPT',  # concept node
        'SENTENCE',  # sentence node
        'CONTEXT_CONCEPT',
        'CONTEXT_SUBJECT',
        'ADJ',  # adjective
        'ADP',  # adposition
        'ADV',  # adverb
        'AUX',  # auxiliary
        'CCONJ',  # coordinating conjunction
        'DET',  # determiner
        'INTJ',  # interjection
        'NOUN',  # noun
        'NUM',  # numeral
        'PART',  # particle
        'PRON',  # pronoun
        'PROPN',  # proper noun
        'PUNCT',  # punctuation
        'SCONJ',  # subordinating conjunction
        'SYM',  # symbol
        'VERB',  # verb
        'X',  # other
    ]

    # Node Source List
    source_list = ['CON', 'SEN', 'BOTH']  # whether a node is from sentence or concept or both

    ner_list = [
        'None',
        '10',  # 人名
        '11',  # 地名
        '12',  # 机构名
        '13',  # 时间
        '14',  # 作品
        '15',  # 食品
        '16',  # 医疗
        '17',  # 工具
        '18',  # 商品
        '19',  # 事件活动
        '20',  # 统称
        '21',  # 特殊物质
        '22',  # 奖项
        '23',  # 生物
        '24',  # 货币
    ]

    # Generate relation dict of index
    rel_list = [
        # https: //universaldependencies.org/u/dep/index.html
        'acl',
        'advcl',
        'advmod',
        'amod',
        'appos',
        'aux',
        'case',
        'cc',
        'ccomp',
        'clf',
        'compound',
        'conj',
        'cop',
        'csubj',
        'dep',
        'det',
        'discourse',
        'dislocated',
        'expl',
        'fixed',
        'flat',
        'goeswith',
        'iobj',
        'list',
        'mark',
        'nmod',
        'nsubj',
        'nummod',
        'obj',
        'obl',
        'orphan',
        'parataxis',
        'punct',
        'reparandum',
        # 'root', # root, not used
        'vocative',
        'xcomp',
        # new defined
        'contains',
        'obtain',
        'talked_in'
    ]
    rel_dict = {}
    index = 0
    for rel in rel_list:
        rel_dict[rel] = index
        index += 1
        rel_dict['-' + rel] = index
        index += 1
    del index, rel

    # embedding configs, used in Model.Embedder
    emb_config = {
        "word": {
            "emb_file_path": DataConfig.EMB_WORD_PATH,  # or None if we not use glove
            "emb_size": None,
            "emb_dim": emb_len_word,
            "need_train": False,
            "need_conv": False,
            "need_emb": True,
            "tag_list": None
        },
        "source": {
            "emb_file_path": DataConfig.EMB_SOURCE_PATH,  # or None if we not use glove
            "emb_size": None,
            "emb_dim": emb_len_source,
            "need_train": True,
            "need_conv": False,
            "need_emb": True,
            "tag_list": source_list
        },
        "pos": {
            "emb_file_path": DataConfig.EMB_POS_PATH,  # or None if we not use glove
            "emb_size": None,
            "emb_dim": emb_len_pos,
            "need_train": True,
            "need_conv": False,
            "need_emb": True,
            "tag_list": pos_list
        },
        "ner": {
            "emb_file_path": DataConfig.EMB_NER_PATH,  # or None if we not use glove
            "emb_size": None,
            "emb_dim": emb_len_ner,
            "need_train": True,
            "need_conv": False,
            "need_emb": True,
            "tag_list": ner_list
        },

    }
    emb_tags = emb_config.keys()


# =========================================================================
# Model Settings
# =========================================================================
# only for the Graph2Graph model
class ModelConfig:
    num_relations = len(EmbConfig.rel_list * 2)
    num_bases = 14
    num_layers = 3


class TrainConfig:
    epoch = 20
    lr = 0.0001
    decay_factor = 0.1
    use_scheduler = True
    num_warm_up = 2


def print_info(con=None):
    print("========================")
    print("Whether use random order data: " + str(DataConfig.USE_RAND_DATA))
    print("Use Scheduler: " + str(TrainConfig.use_scheduler))
    if con is None:
        print("ModelConfig")
        print("    num_relations: %s" % ModelConfig.num_relations)
        print("    num_bases: %s" % ModelConfig.num_bases)
        print("    num_layers: %s" % ModelConfig.num_layers)

        print("TrainConfig")
        print("    epoch: %s" % TrainConfig.epoch)
        print("    lr: %s" % TrainConfig.lr)
        print("    decay_factor: %s" % TrainConfig.decay_factor)
        print("    num_warm_up: %s" % TrainConfig.num_warm_up)
    else:
        print("ModelConfig")
        print("    num_relations: %s" % con.ModelConfig.num_relations)
        print("    num_bases: %s" % con.ModelConfig.num_bases)
        print("    num_layers: %s" % con.ModelConfig.num_layers)

        print("TrainConfig")
        print("    epoch: %s" % con.TrainConfig.epoch)
        print("    lr: %s" % con.TrainConfig.lr)
        print("    decay_factor: %s" % con.TrainConfig.decay_factor)
        print("    num_warm_up: %s" % con.TrainConfig.num_warm_up)
    print("========================")
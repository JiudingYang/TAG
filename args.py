import argparse
from configs import *

# parser used to read argument
parser = argparse.ArgumentParser(description='xxxxx')

# ==================================
#            Universal
# ==================================
parser.add_argument(
    '--seed', type=int, default=UNIVERSAL_SEED)
parser.add_argument(
    '--device', type=str, default=DEVICE, help='device to run on')
# ==================================
#          Preprocessing
# ==================================
parser.add_argument(
    '--rand_data', type=bool, default=DataConfig.USE_RAND_DATA,
    help='whether use random data to train')
parser.add_argument(
    '--tag_emb_generated', type=bool, default=DataConfig.IS_TAGEMB_GENERATED,
    help='whether tag emb matrix is generated')
parser.add_argument(
    '--word_dictcut', type=bool, default=DataConfig.IS_DICT_CUT,
    help='whether a word dict is generated from original glove')
parser.add_argument(
    '--preprocessed', type=bool, default=DataConfig.IS_PREPROCESSED, help='whether is preprocessed')

# ==================================
#      Model & Training  Settings
# ==================================
parser.add_argument(
    '--nr', type=int, default=ModelConfig.num_relations, help='num of relations of rgcn')
parser.add_argument(
    '--nb', type=int or list, default=ModelConfig.num_bases, help='num of bases of rgcn')
parser.add_argument(
    '--nl', type=int or list, default=ModelConfig.num_layers, help='num of layers of rgcn')
parser.add_argument(
    '--epoch', type=int, default=TrainConfig.epoch, help='num of epoch')
parser.add_argument(
    '--lr', type=int, default=TrainConfig.lr, help='initial learning rate')
parser.add_argument(
    '--df', type=int, default=TrainConfig.decay_factor, help='an the end of training, lr decays to certain '
                                                             'ratio, defalut 0.1')
# parser.add_argument(
#     '--br', type=int, default=TrainConfig.balance_ratio, help='set the ration of pos and neg data in Training '
#                                                               'set')


def update_configs(_configs, _args):
    # update configs according to input args
    _configs.UNIVERSAL_SEED = _args.seed
    _configs.DEVICE = torch.device(_args.device)

    _configs.DataConfig.IS_TAGEMB_GENERATED = _args.tag_emb_generated
    _configs.DataConfig.IS_DICT_CUT = _args.word_dictcut
    _configs.DataConfig.IS_PREPROCESSED = _args.preprocessed

    _configs.DataConfig.USE_RAND_DATA = _args.rand_data

    _configs.ModelConfig.num_relations = _args.nr
    _configs.ModelConfig.num_bases = _args.nb
    _configs.ModelConfig.num_layers = _args.nl
    _configs.TrainConfig.epoch = _args.epoch
    _configs.TrainConfig.lr = _args.lr
    _configs.TrainConfig.decay_factor = _args.df
    # _configs.TrainConfig.balance_ratio = _args.br

    if not _configs.DataConfig.USE_RAND_DATA:
        _configs.DataConfig.PREPRO_TRAIN_DATA = 'prepro_train_set.dat'
        _configs.DataConfig.PREPRO_TEST_DATA = 'prepro_test_set.dat'
        _configs.DataConfig.TRAIN_DATA = 'train_set.json'
        _configs.DataConfig.TEST_DATA = 'test_set.json'
    else:
        _configs.DataConfig.PREPRO_TRAIN_DATA = 'prepro_train_set_rand.dat'
        _configs.DataConfig.PREPRO_TEST_DATA = 'prepro_test_set_rand.dat'
        _configs.DataConfig.TRAIN_DATA = 'train_set_rand.json'
        _configs.DataConfig.TEST_DATA = 'test_set_rand.json'

    _configs.DataConfig.PREPRO_TRAIN_PATH = _configs.DataConfig.PROCESSED_DATA_FOLDER + _configs.DataConfig.PREPRO_TRAIN_DATA
    _configs.DataConfig.PREPRO_TEST_PATH = _configs.DataConfig.PROCESSED_DATA_FOLDER + _configs.DataConfig.PREPRO_TEST_DATA
    _configs.DataConfig.TRAIN_DATA_PATH = _configs.DataConfig.RAW_DATA_FOLDER + _configs.DataConfig.TRAIN_DATA
    _configs.DataConfig.TEST_DATA_PATH = _configs.DataConfig.RAW_DATA_FOLDER + _configs.DataConfig.TEST_DATA

    return _configs


import args
from args import *
import configs as ori_configs
from Trainer.Graph_Trainer import *
import torch
from Model.GraphModel import TAG
from UtilPack.DataUtil import load_embs
from UtilPack.ExpUtil import set_random_seed
# set a default seed right after all imports
import time
set_random_seed(999)


def main(args):
    configs = update_configs(ori_configs, args)
    DataConfig = configs.DataConfig
    TrainConfig = configs.TrainConfig
    EmbConfig = configs.EmbConfig
    DEVICE = configs.DEVICE
    ModelConfig = configs.ModelConfig
    set_random_seed(UNIVERSAL_SEED)

    print('===============================')
    print('             TAG')
    print('===============================')
    # check whether we need preprocessing data
    if not DataConfig.IS_PREPROCESSED:
        from DataProcess.DataLoader import pre_process_data
        start = time.clock()
        pre_process_data(DataConfig.TRAIN_DATA_PATH, DataConfig.PREPRO_TRAIN_PATH, DEVICE)
        pre_process_data(DataConfig.TEST_DATA_PATH, DataConfig.PREPRO_TEST_PATH, DEVICE)
        elapsed = (time.clock() - start)
        print("Preprocessing Time: %s sec" % elapsed)

    # change int to list to fit for loops
    if type(ModelConfig.num_layers) != list:
        ModelConfig.num_layers = [ModelConfig.num_layers]
    if type(ModelConfig.num_bases) != list:
        ModelConfig.num_bases = [ModelConfig.num_bases]

    nl_list = ModelConfig.num_layers
    nb_list = ModelConfig.num_bases

    Best_Model = None
    Best_Para = None
    best_validation_f1 = 0
    Model = None
    # train all parameter combinations
    best_setting = [0, 0]
    for nl in nl_list:
        ModelConfig.num_layers = nl
        for nb in nb_list:
            start = time.clock()
            ModelConfig.num_bases = nb
            F1, Para, Model = TAG_Trainer(configs)
            elapsed = (time.clock() - start)
            print("Training Time: %s sec" % elapsed)
            print("========================")
            if best_validation_f1 < F1:
                best_validation_f1 = F1
                Best_Para = Para.copy()
                best_setting = [nb, nl]

    # eval Test set
    ModelConfig.num_layers = best_setting[1]
    ModelConfig.num_bases = best_setting[0]
    emb_dict, emb_mats = load_embs(EmbConfig.emb_config)
    Best_Model = TAG(emb_dict, emb_mats).to(DEVICE)
    Best_Model.load_state_dict(Best_Para)

    graph_f1, graph_acc, _ = TAG_Test(Best_Model, configs=configs)
    print('================TEST================')
    print('Best layers and bases are: %s:%s' % (best_setting[0], best_setting[1]))
    print('Test F1 for TAG: ' + str(graph_f1 * 100))
    print('Test ACC for TAG: ' + str(graph_acc * 100))
    torch.save(Best_Model.to('cpu').state_dict(), DataConfig.BEST_PARA_PATH)


if __name__ == '__main__':
    main(parser.parse_args())

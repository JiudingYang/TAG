import random
import torch
import torch.nn as nn
import torch.optim as optim
import configs as ori_configs
import _pickle as pk
from Model.GraphModel import TAG
from UtilPack.DataUtil import load_embs
from sklearn.metrics import f1_score, accuracy_score
from DataProcess.DLUtils import gen_unique_set
import math
import transformers as trans


def TAG_Trainer(configs=None):
    if configs is None:
        configs = ori_configs

    DataConfig = configs.DataConfig
    TrainConfig = configs.TrainConfig
    EmbConfig = configs.EmbConfig
    DEVICE = configs.DEVICE
    ModelConfig = configs.ModelConfig

    with open(DataConfig.PREPRO_TRAIN_PATH, 'rb') as f:
        Data = pk.load(f)

    print('Training Information')
    configs.print_info()

    total = len(Data)
    print('Length of the Training Set: ' + str(total))

    emb_dict, emb_mats = load_embs(EmbConfig.emb_config)
    Graph_Model = TAG(emb_dict, emb_mats, configs=configs).to(DEVICE)
    random.shuffle(Data)
    if DataConfig.USE_RAND_DATA:
        # train set and valid set
        cut = int(total * 0.875)
        Train_Data = Data[:cut]
        Valid_Data = Data[cut:]
    else:
        Train_Data, Valid_Data = gen_unique_set(Data, 0.125)
    del Data
    train_data_num = int(total*0.875)
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    #decay = 1 - pow(TrainConfig.decay_factor, 1 / (train_data_num * TrainConfig.epoch))
    decay = 0.0001
    optimizer = optim.Adam(Graph_Model.parameters(), eps=1e-8, lr=TrainConfig.lr, weight_decay=decay)
    # schedule learning rate
    scheduler = None
    if TrainConfig.use_scheduler:

        scheduler = trans.optimization.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(TrainConfig.num_warm_up * train_data_num),
            num_training_steps=TrainConfig.epoch * train_data_num
        )

    valid_best_acc = 0
    valid_best_f1 = 0

    best_para = Graph_Model.state_dict().copy()

    for epoch in range(TrainConfig.epoch):

        random.shuffle(Train_Data)
        Graph_Model.train()

        for index, row in enumerate(Train_Data):
            data = row['graph_data']
            edge_type = data.rel.to(DEVICE)
            edge_index = data.edge_index.to(DEVICE)
            emb_idx_dict = {'source': data.source_tag.clone().detach().to(DEVICE),
                            'pos': data.pos_tag.clone().detach().to(DEVICE),
                            'ner': data.ner_tag.clone().detach().to(DEVICE)}

            x = data.x.to(DEVICE).float()

            y = Graph_Model(x, emb_idx_dict, edge_index, edge_type)
            optimizer.zero_grad()
            loss = criterion(y, torch.tensor([row['label']]).float().to(DEVICE))
            loss.backward()
            optimizer.step()
            if TrainConfig.use_scheduler:
                scheduler.step()
            # if index % 200 == 1 or index == len(Train_Data):
            #    print("\r   Processing  %8s / %-8s : %4s " % (index, len(Train_Data), loss.data))

        valid_f1, valid_acc, valid_loss = TAG_Test(Graph_Model, valid_set=Valid_Data, configs=configs)
        train_f1, train_acc, train_loss = TAG_Test(Graph_Model, valid_set=Train_Data, configs=configs)

        valid_best_acc = max(valid_best_acc, valid_acc)
        if valid_best_f1 < valid_f1:
            best_para = Graph_Model.state_dict().copy()
            valid_best_f1 = valid_f1

        print('\r')
        print("Epoc: %s " % (epoch + 1))
        print("Train F1:  %s|Train Acc:  %s|Train Loss:  %s  " % (train_f1, train_acc, train_loss))
        print("Valid F1:  %s|Valid Acc:  %s|Valid Loss:  %s  " % (valid_f1, valid_acc, valid_loss))
    print('\rGRAPH:BEST_VALID')
    print('Best Valid F1 for TAG: ' + str(valid_best_f1 * 100))
    print('Best Valid ACC for TAG: ' + str(valid_best_acc * 100))
    torch.save(best_para, DataConfig.MODEL_PARA_FOLDER + 'nl-%s_nb-%s_BestPara.mod' % (
        ModelConfig.num_layers, ModelConfig.num_bases))

    Graph_Model.load_state_dict(best_para)
    graph_f1, graph_acc, _ = TAG_Test(Graph_Model, configs=configs)
    print('\rGRAPH:TEST')
    print('Test F1 for TAG: ' + str(graph_f1 * 100))
    print('Test ACC for TAG: ' + str(graph_acc * 100))

    return valid_best_f1, best_para, Graph_Model


def TAG_Test(model, valid_set=None, configs=None):
    if configs is None:
        configs = ori_configs

    DataConfig = configs.DataConfig
    TrainConfig = configs.TrainConfig
    EmbConfig = configs.EmbConfig
    DEVICE = configs.DEVICE
    ModelConfig = configs.ModelConfig

    if valid_set is None:
        # load test set
        with open(DataConfig.PREPRO_TEST_PATH, 'rb') as f:
            Test_Data = pk.load(f)
    else:
        Test_Data = valid_set

    model.eval()
    graph_pred = []
    true_pred = []
    loss = 0
    loss_fcn = nn.BCEWithLogitsLoss().to(DEVICE)
    with torch.no_grad():
        for row in Test_Data:
            data = row['graph_data']
            edge_type = data.rel.to(DEVICE)
            edge_index = data.edge_index.to(DEVICE)
            emb_idx_dict = {'source': data.source_tag.clone().detach().to(DEVICE),
                            'pos': data.pos_tag.clone().detach().to(DEVICE),
                            'ner': data.ner_tag.clone().detach().to(DEVICE)}
            x = data.x.to(DEVICE)
            try:
                y_pred = model(x, emb_idx_dict, edge_index, edge_type)
            except:
                print(row['concept'], row['sentence'])

            loss += loss_fcn(y_pred, torch.tensor([row['label']]).float().to(DEVICE))

            y_pred = torch.sigmoid(y_pred)
            if y_pred >= 0.5:
                y_pred = 1
            else:
                y_pred = 0
            graph_pred.append(y_pred)
            true_pred.append(row['label'])

    graph_f1 = f1_score(true_pred, graph_pred)
    graph_acc = accuracy_score(true_pred, graph_pred)

    return graph_f1, graph_acc, loss

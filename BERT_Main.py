import configs as ori_configs
from torch.utils.data import DataLoader, TensorDataset
from Model.matching import FT_Match
from sklearn.metrics import f1_score, accuracy_score
import transformers as trans
import _pickle as pk
import torch
import torch.nn as nn
import random
from UtilPack.ExpUtil import set_random_seed
from DataProcess.DLUtils import gen_unique_set
set_random_seed(999)

lr = 0.000001
total_epoch = 30
batch_size = 8
max_length = 256
print("Training Parameters:")
print("lr: %s|batch_size: %s" % (lr, batch_size))
print('use_scheduler: ' + str(ori_configs.TrainConfig.use_scheduler))
print('warm_up: ' + str(ori_configs.TrainConfig.num_warm_up))
print('Rand_Data:'+str(ori_configs.DataConfig.USE_RAND_DATA))


def load_data(data):
    concept = []
    sentence = []
    label_list = []
    for row in data:
        concept.append(row['concept'])
        sentence.append(row['sentence'])
        label_list.append(row['label'])
    tokenizer = trans.BertTokenizer.from_pretrained('bert-base-chinese')
    concept_data = tokenizer.batch_encode_plus(concept, max_length=max_length, pad_to_max_length=True)
    sentence_data = tokenizer.batch_encode_plus(sentence, max_length=max_length, pad_to_max_length=True)
    out_data = torch.tensor([concept_data['input_ids'], concept_data['attention_mask'],
                             sentence_data['input_ids'], sentence_data['attention_mask']], )

    return TensorDataset(torch.transpose(out_data, 0, 1), torch.tensor(label_list))


@torch.enable_grad()
def train(batch, configs=None):
    if configs is None:
        configs = ori_configs

    DataConfig = configs.DataConfig
    TrainConfig = configs.TrainConfig
    EmbConfig = configs.EmbConfig
    DEVICE = configs.DEVICE
    ModelConfig = configs.ModelConfig

    with open(DataConfig.PREPRO_TRAIN_PATH, 'rb') as f:
        Data = pk.load(f)

    total = len(Data)
    print('Length of the balanced data: ' + str(total))
    random.shuffle(Data)
    # cut data to train and text
    if DataConfig.USE_RAND_DATA:
        cut = int(total * 0.875)
        Train_Data = Data[:cut]
        Valid_Data = Data[cut:]
    else:
        Train_Data, Valid_Data = gen_unique_set(Data, 0.125)
    train_dataset = load_data(Train_Data)
    valid_dataset = load_data(Valid_Data)
    del Data, Train_Data, Valid_Data

    print('===============================')
    print('        Text Training')
    print('===============================')

    model = FT_Match(0).to(DEVICE)
    model_state = model.state_dict().copy()

    criterion = nn.BCEWithLogitsLoss(reduction='mean').to(DEVICE)
    optimizer = trans.optimization.AdamW(params=model.parameters(), lr=lr)
    # use scheduler
    scheduler = None
    if TrainConfig.use_scheduler:
        scheduler = trans.optimization.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(TrainConfig.num_warm_up*(7000/batch_size)),
            num_training_steps=total_epoch*(7000/batch_size)
        )

    text_best_acc = 0
    text_best_f1 = 0
    # for item in Train_Data:
    dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=False)
    for epoch in range(total_epoch):
        model.train()
        index = 0
        for i in dataloader:
            index += 1
            y = i[1].float().to(DEVICE)
            y_pred = model(torch.transpose(i[0], 0, 1).to(DEVICE))
            optimizer.zero_grad()
            loss = criterion(y_pred.squeeze(1), y)
            loss.backward()
            optimizer.step()
            if TrainConfig.use_scheduler:
                scheduler.step()
            #if index%50 == 1 or index == len(dataloader):
                #print("\r   Processing  %8s / %-13s : %4s" % (index, len(dataloader), loss))

        valid_f1, valid_acc, _ = test(model, valid_dataset=valid_dataset, configs=configs)
        train_f1, train_acc, train_loss = test(model, valid_dataset=train_dataset, configs=configs)
        # store best model's parameters
        if text_best_f1 < valid_f1:
            model_state = model.state_dict().copy()
        # update best values
        text_best_acc = max(valid_acc, text_best_acc)
        text_best_f1 = max(valid_f1, text_best_f1)

        print('============TEXT=============')
        print("Epoc:  " + str(epoch + 1))
        print("Train F1:  %s " % train_f1)
        print("Train Acc:  %s " % train_acc)
        print("Train Loss:  %s " % train_loss)
        print("Valid F1: %s " % valid_f1)
        print("Valid Acc:  %s " % valid_acc)

    torch.save(model_state, 'Data/Model_Para/Bert_Para')
    print('Best F1 for Valid: ' + str(text_best_f1 * 100))
    print('Best ACC for Valid: ' + str(text_best_acc * 100))
    model.load_state_dict(model_state)
    return text_best_f1, model_state, model


def test(model, valid_dataset=None, configs=None):
    if configs is None:
        configs = ori_configs

    DataConfig = configs.DataConfig
    TrainConfig = configs.TrainConfig
    EmbConfig = configs.EmbConfig
    DEVICE = configs.DEVICE
    ModelConfig = configs.ModelConfig

    if valid_dataset is None:
        # load test set
        with open(DataConfig.PREPRO_TEST_PATH, 'rb') as f:
            Test_Data = pk.load(f)
        test_dataset = load_data(Test_Data)

    else:
        test_dataset = valid_dataset

    model.eval()
    loss = 0
    true = []
    pred = []
    with torch.no_grad():
        # TODO: change batch size to 1 for testing, save mem
        for index, row in enumerate(test_dataset.tensors[0]):
            x1, x1t, x2, x2t = row.to(DEVICE)
            x1, x1t, x2, x2t = x1.unsqueeze(0), x1t.unsqueeze(0), x2.unsqueeze(0), x2t.unsqueeze(0)
            y = test_dataset.tensors[1][index].float()

            y_pred = model([x1, x1t, x2, x2t]).squeeze(1).to('cpu')
            loss += nn.BCEWithLogitsLoss(reduction='mean')(y_pred.squeeze(0), y)
            y_pred = torch.sigmoid(y_pred).round().squeeze(0)
            true.append(y)
            pred.append(y_pred)

        text_acc = accuracy_score(true, pred)
        text_f1 = f1_score(true, pred)

    return text_f1, text_acc, loss


if __name__ == '__main__':
    set_random_seed(ori_configs.UNIVERSAL_SEED)
    f1=0
    acc=0
    for i in range(5):    
        valid_best_f1, model_state, model = train(batch_size)
        text_f1, text_acc, _ = test(model)
        f1+=text_f1
        acc+=text_acc
        print('Best F1 for Test: ' + str(text_f1 * 100))
        print('Best ACC for Test: ' + str(text_acc * 100))
    print('Avg F1 for Test: ' + str(f1/5 * 100))
    print('Avg ACC for Test: ' + str(acc/5 * 100))


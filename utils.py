import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import roc_auc_score, log_loss
BATCH_SIZE = 128
def get_data(filename):
    # read criteo dateset, split train and test
    num_cols = ["I"+str(i) for i in range(1, 14)]
    cat_cols = ['C'+str(i) for i in range(1,27)]
    # read file
    col_names_train = ['label'] + ["I"+str(i) for i in range(1, 14)] + ['C'+str(i) for i in range(1,27)]
    df_train = pd.read_csv(filename, sep=',')
    df_train = df_train.iloc[:5000]# 小数据测试
    data = df_train
    labels = data['label']
    del data['label']
    train_len = int(data.shape[0]*0.9)
    # prepocess
    emb_col = dict()# 需要emb的fea和类别数(>256) 稀疏特征embedding
    for i, col in enumerate(data.columns):
        if col in num_cols:
            data[col] = data[col].fillna(data[col].median())
            data[col] = data[col].apply(lambda x: np.log(x + 1) if x > -1 else -1)# 平滑

        if col in cat_cols:
            data[col] = data[col].fillna('null')
            data[col] = data[col].astype('str')
            data[col] = LabelEncoder().fit_transform(data[col])
            # find category>256, need embedding
            # if data[col].nunique() > 256:
            emb_col[i] = data[col].nunique()# 列索引，类别数
    unembed_col = [i for i in range(len(data.columns)) if i not in emb_col]
    # split train and test
    idxes = np.arange(data.shape[0])
    #np.random.seed(2021)   
    #np.random.shuffle(idxes)
    y_train, y_test = labels.iloc[idxes[:train_len]].values, labels.iloc[idxes[train_len:]].values
    x_train, x_test = data.iloc[idxes[:train_len]].values, data.iloc[idxes[train_len:]].values
    train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
    return train_loader, test_loader, emb_col, unembed_col

def evaluate(model, test_loader):
    # acc valid
    model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for idx, (x, y) in (enumerate(test_loader)):
            y_pred = model(x.cuda())# 为1的概率
            y_pred = y_pred.detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
            val_true.extend(y.squeeze().cpu().numpy().tolist())
    
    return log_loss(val_true, val_pred), roc_auc_score(val_true, val_pred)
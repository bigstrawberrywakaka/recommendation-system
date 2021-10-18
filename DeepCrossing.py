# deep crossing
# paper:
# 	https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf
# data:
#   https://www.kaggle.com/mrkmakr/criteo-dataset
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# args
embed_dim = 8
BATCH_SIZE = 256
epoch = 5
hidden_layers = [512,256,128,64,32]
class residual_block(nn.Module):
    def __init__(self, stack_dim, hidden_dim):# emb_col是一个词典，对应的值是特征总类别数
        super(residual_block, self).__init__()
        self.fc1 = nn.Linear(stack_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, stack_dim, bias=True)
    def forward(self, x):
        return torch.relu(self.fc2(torch.relu(self.fc1(x))) + x)# 拟合残差

class DeepCrossing(nn.Module):
    def __init__(self, emb_col, unembed_col):# emb_col是一个词典，对应的值是特征总类别数
        super(DeepCrossing, self).__init__()
        self.emb_col = emb_col
        self.unembed_col = unembed_col
        self.stack_dim = embed_dim * len(self.emb_col) + len(self.unembed_col)
        self.embedding_layers = nn.ModuleList([nn.Embedding(self.emb_col[fea], embed_dim) for fea in self.emb_col])
        self.residual_layers = [residual_block(self.stack_dim, hidden_dim).cuda() for hidden_dim in hidden_layers]# >256的稀疏fea个数
        self.fc = nn.Linear(self.stack_dim, 1)
    def forward(self, x):
        embedding_col = [self.embedding_layers[i](x[:, col]) for i, col in enumerate(self.emb_col)]
        unembedding_col = x[:,self.unembed_col]
        embedding_col = torch.cat(embedding_col, axis=-1)
        output = torch.cat([embedding_col, unembedding_col], axis=-1)
        for residual_layer in self.residual_layers:
            output = residual_layer(output)
        output = torch.sigmoid(self.fc(output)).squeeze(-1)
        return output

def get_data(filename):
    num_cols = ["I"+str(i) for i in range(1, 14)]
    cat_cols = ['C'+str(i) for i in range(1,27)]
    # read file
    col_names_train = ['Label'] + ["I"+str(i) for i in range(1, 14)] + ['C'+str(i) for i in range(1,27)]
    df_train = pd.read_csv(filename, sep='\t', names=col_names_train, chunksize=100000) # ten chunks: first 1,000,000
    data = df_train.get_chunk(2500000)# train 2000000, test 500000
    labels = data['Label']
    del data['Label']
    # prepocess
    emb_col = dict()# 需要emb的fea和类别数(>256) 稀疏特征embedding
    for i, col in enumerate(data.columns):
        if col in num_cols:
            data[col] = data[col].fillna(data[col].median())
        if col in cat_cols:
            data[col] = data[col].fillna('null')
            data[col] = data[col].astype('str')
            data[col] = LabelEncoder().fit_transform(data[col])
            # find category>256, need embedding
            if data[col].nunique() > 256:
                emb_col[i] = data[col].nunique()# 列索引，类别数
    unembed_col = [i for i in range(len(data.columns)) if i not in emb_col]
    # split train and test
    idxes = np.arange(data.shape[0])
    np.random.seed(2021)   
    np.random.shuffle(idxes)
    y_train, y_test = labels.iloc[idxes[:2000000]].values, labels.iloc[idxes[2000000:]].values
    x_train, x_test = data.iloc[idxes[:2000000]].values, data.iloc[idxes[2000000:]].values
    train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
    return train_loader, test_loader, emb_col, unembed_col

def evaluate(model, test_loader):
    model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for idx, (x, y) in (enumerate(test_loader)):
            y_pred = model(x.cuda())# 为1的概率
            y_pred = y_pred.detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
            val_true.extend(y.squeeze().cpu().numpy().tolist())
    
    return roc_auc_score(val_true, val_pred)  

if __name__ == '__main__':
    filename = './data/dac/train.txt'
    train_loader, test_loader, emb_col, unembed_col = get_data(filename)
    # train...
    model = DeepCrossing(emb_col, unembed_col)
    model.cuda()
    total_loss = 0.0
    critetion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for i in range(epoch):
        for idx, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            y_pred = model(x)
            loss = critetion(y_pred, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (idx + 1) % (len(train_loader)//5) == 0:    # 只打印五次结果
                print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f}".format(
                          i+1, idx+1, len(train_loader), total_loss/(idx+1)))

        # valid...
        model.eval()
        auc = evaluate(model, test_loader)
        print("epoch ", i, "auc is: ", auc)
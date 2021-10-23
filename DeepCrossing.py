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
from utils import get_data, evaluate
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import warnings
warnings.filterwarnings('ignore')

# args
embed_dim = 32
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

if __name__ == '__main__':
    filename = './data/criteo_sampled_data.csv'
    train_loader, test_loader, emb_col, unembed_col = get_data(filename)
    # train...
    model = DeepCrossing(emb_col, unembed_col)
    model.cuda()
    total_loss = 0.0
    oof_auc = 0.0
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
        oof_auc += auc
    print("average valid auc: ", oof_auc/epoch)
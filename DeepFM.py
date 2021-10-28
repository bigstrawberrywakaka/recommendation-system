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
embed_dim = 8
epoch = 5
hidden_dim = 128
k=30

class FM(nn.Module):
    def __init__(self, emb_col, unembed_col, k):# k<<stack_dim
        super(FM, self).__init__()
        self.stack_dim = len(emb_col) * embed_dim + len(unembed_col)
        self.k = k
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w1 = nn.Parameter(nn.init.xavier_uniform_(torch.randn(self.stack_dim, 1)))
        self.v = nn.Parameter(nn.init.xavier_uniform_(torch.randn(self.stack_dim, self.k)))
        #self.fc = nn.Linear(self.k, 1)

    def forward(self, x):
        x = x.float()
        linear = self.w0 + torch.mm(x, self.w1)
        a = torch.pow(torch.mm(x, self.v), 2)# (bs, k)
        b = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))
        cross = 0.5 * (a - b)
        fm_output = linear + cross
        return fm_output

class DNN(nn.Module):
    def __init__(self, emb_col, unembed_col):
        super(DNN, self).__init__()
        self.stack_dim = len(emb_col)*embed_dim + len(unembed_col)
        self.fc1 = nn.Linear(self.stack_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, k)
    def forward(self, x):
        DNN_output = torch.relu(self.fc2(torch.relu(self.fc1(x))))
        return DNN_output

class DeepFM(nn.Module):
    def __init__(self, emb_col, unembed_col):
        super(DeepFM, self).__init__()
        self.emb_col = emb_col
        self.unembed_col = unembed_col
        self.stack_dim = embed_dim * len(self.emb_col) + len(self.unembed_col)
        self.embedding_layers = nn.ModuleList([nn.Embedding(self.emb_col[fea], embed_dim) for fea in self.emb_col])
        FM_layers = FM(emb_col, unembed_col, 30)
        DNN_layers = DNN(emb_col, unembed_col)
        self.FM_layers = FM_layers.cuda()
        self.DNN_layers = DNN_layers.cuda()
        self.fc_final = nn.Linear(k, 1)
    def forward(self, x):
        embedding_col = [self.embedding_layers[i](x[:, col]) for i, col in enumerate(self.emb_col)]
        unembedding_col = x[:,self.unembed_col]
        embedding_col = torch.cat(embedding_col, axis=-1)
        data = torch.cat([embedding_col, unembedding_col], axis=-1)
        FM_output = self.FM_layers(data)
        DNN_output = self.DNN_layers(data)
        output = torch.sigmoid(self.fc_final(torch.add(FM_output, DNN_output)))
        return output

if __name__ == '__main__':
    filename = './data/criteo_sampled_data.csv'
    train_loader, test_loader, emb_col, unembed_col = get_data(filename)
    # train...
    model = DeepFM(emb_col, unembed_col)
    model.cuda()
    total_loss, oof_auc, valid_loss = 0.0, 0.0, 0.0
    critetion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for i in range(epoch):
        for idx, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            y_pred = model(x).squeeze(-1)
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
        logloss, auc = evaluate(model, test_loader)
        print("epoch ", i, "auc is: ", auc, "logloss is: ", logloss)
        oof_auc += auc
        valid_loss += logloss
    print(epoch , "epoch average valid auc: ", oof_auc/epoch, "logloss is: ", valid_loss/epoch)
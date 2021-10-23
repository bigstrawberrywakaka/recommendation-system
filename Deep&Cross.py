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

BATCH_SIZE = 128
embed_dim = 8
epoch = 5
cross_layer_len = 4
dnn_dim = [128, 128, 128]

class cross_network(nn.Module):
    def __init__(self, stack_dim):
        super(cross_network, self).__init__()
        weight = nn.init.xavier_uniform_(torch.randn(stack_dim, 1))
        self.weight = nn.Parameter(weight)
        bias = nn.init.zeros_(torch.randn(stack_dim))
        self.bias = nn.Parameter(bias)
    def forward(self, x0, x):# xl+1 = x0 * xl.T * w + b + xl
        emb_dim = x.shape[-1]
        x_w = x.reshape(-1, 1, emb_dim)
        x_w = torch.matmul(x_w, self.weight).squeeze(-1)# (bs, 1)是个标量
        cross = x0 * x_w
        return cross + self.bias + x
        

class deep_network(nn.Module):
    def __init__(self, stack_dim):
        super(deep_network, self).__init__()
        self.fc1 = nn.Linear(stack_dim, dnn_dim[0])
        self.fc2 = nn.Linear(dnn_dim[0], dnn_dim[1])
        self.fc3 = nn.Linear(dnn_dim[1], dnn_dim[2])
    def forward(self, x):
        dnn_output = torch.relu(self.fc1(x))
        dnn_output = torch.relu(self.fc2(dnn_output))
        dnn_output = torch.relu(self.fc3(dnn_output))
        return dnn_output

class DCN(nn.Module):
    def __init__(self, emb_col, unembed_col):# emb_col是一个词典，对应的值是特征总类别数
        super(DCN, self).__init__()
        self.emb_col = emb_col
        self.unembed_col = unembed_col
        self.embedding_layers = nn.ModuleList([nn.Embedding(self.emb_col[fea], embed_dim, norm_type=2) for fea in self.emb_col])
        self.stack_dim = len(emb_col) * embed_dim + len(unembed_col)
        self.cross_layers = nn.ModuleList(cross_network(self.stack_dim) for _ in range(cross_layer_len))
        self.dnn_layers = deep_network(self.stack_dim)
        self.final_dim = self.stack_dim + dnn_dim[-1]
        self.fc = nn.Linear(self.final_dim, 1)
    def forward(self, x):
        embedding_col = [self.embedding_layers[i](x[:, col]) for i, col in enumerate(self.emb_col)]
        unembedding_col = x[:,self.unembed_col]
        embedding_col = torch.cat(embedding_col, axis=-1)
        output = torch.cat([embedding_col, unembedding_col], axis=-1)
        # cross_net
        cross_output = output#.unsqueeze(2)
        x0 = cross_output
        for cross_layer in self.cross_layers:
            cross_output = cross_layer(x0, cross_output)
        cross_output = cross_output.squeeze(-1)# (bs, stack_dim)
        # deep_net
        dnn_output = self.dnn_layers(output)
        output = torch.cat([cross_output, dnn_output], axis=-1)
        #print(output.shape)
        output = torch.sigmoid(self.fc(output))
        return output

if __name__ == '__main__':
    filename = './data/criteo_sampled_data.csv'
    train_loader, test_loader, emb_col, unembed_col = get_data(filename)
    # train...
    model = DCN(emb_col, unembed_col)
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
        logloss, auc = evaluate(model, test_loader)# 这是所有验证集的吧
        print("epoch ", i, "auc is: ", auc, "logloss is: ", logloss)
        oof_auc += auc
        valid_loss += logloss
    print(epoch , "epoch average valid auc: ", oof_auc/epoch, "auc: logloss is: ", valid_loss/epoch)

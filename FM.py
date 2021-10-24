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

epoch = 5

class FM(nn.Module):
    def __init__(self, emb_col, unembed_col, k=10):# k<<stack_dim
        super(FM, self).__init__()
        self.stack_dim = len(emb_col) + len(unembed_col)
        self.k = k
        self.fc1 = nn.Linear(self.stack_dim, 1)
        self.fc2 = nn.Linear(self.k, 1)
        self.v = nn.Parameter(nn.init.xavier_uniform_(torch.randn(self.stack_dim, self.k)))

    def forward(self, x):
        x = x.float()
        linear = self.fc1(x)
        a = torch.pow(torch.mm(x, self.v), 2)# (bs, k)
        b = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))
        cross = 0.5 * (a - b)
        fm_output = linear + cross
        return torch.sigmoid(self.fc2(fm_output))

if __name__ == '__main__':
    filename = './data/criteo_sampled_data.csv'
    train_loader, test_loader, emb_col, unembed_col = get_data(filename)
    # train...
    model = FM(emb_col, unembed_col)
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
    print(epoch , "epoch average valid auc: ", oof_auc/epoch, "auc: logloss is: ", valid_loss/epoch)
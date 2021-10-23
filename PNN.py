# PNN
# paper:
# 	https://arxiv.org/pdf/1611.00144.pdf
# data:
#   https://www.kaggle.com/mrkmakr/criteo-dataset
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import get_data, evaluate
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import warnings
warnings.filterwarnings('ignore')

# args
embed_dim = 48
BATCH_SIZE = 256
epoch = 5
product_type = "Outer"# Inner or Outer
hidden_dim = 64

class PNN(nn.Module):
	def __init__(self, emb_col, unembed_col, product_type, product_layer_dim=10):
		super(PNN, self).__init__()
		self.emb_col = emb_col
		self.unembed_col = unembed_col
		self.product_layer_dim = product_layer_dim# paper中的D1
		self.embedding_layers = nn.ModuleList([nn.Embedding(self.emb_col[fea], embed_dim) for fea in self.emb_col])
		lz_weight = torch.randn((self.product_layer_dim, len(self.emb_col), embed_dim))# D1*N*M
		lz_weight = nn.init.xavier_uniform_(lz_weight)# 让随机的权重服从均匀分布
		self.lz_weight = nn.Parameter(lz_weight)
		self.product_type = product_type
		if self.product_type == 'Inner':
			theta = torch.randn((self.product_layer_dim, len(self.emb_col), embed_dim))
			theta = nn.init.xavier_uniform_(theta)# 让随机的权重服从均匀分布
			self.theta = nn.Parameter(theta)
		else:
			lp_weight = torch.randn((self.product_layer_dim, embed_dim, embed_dim))
			lp_weight = nn.init.xavier_uniform_(lp_weight)# 让随机的权重服从均匀分布
			self.lp_weight = nn.Parameter(lp_weight)
		self.lp_dim = self.product_layer_dim
		self.lz_dim = self.product_layer_dim
		self.dnn_dim = self.lz_dim + self.lp_dim + len(unembed_col)
		self.fc1 = nn.Linear(self.dnn_dim, hidden_dim, bias=True)
		self.fc2 = nn.Linear(hidden_dim, 1)
	def forward(self, x):
		embedding_col = [self.embedding_layers[i](x[:, col]) for i, col in enumerate(self.emb_col)]
		embedding_col = torch.cat(embedding_col, axis=-1).reshape(-1, len(self.emb_col), embed_dim)
		lz = torch.einsum("bnm, dnm->bd", embedding_col, self.lz_weight)# (batch_size, product_layer_dim)
		unembedding_col = x[:,self.unembed_col]
		if self.product_type == 'Inner':
			delta = torch.einsum("bnm, dnm->bdnm", embedding_col, self.theta)
			lp = torch.einsum("bdnm, bdnm->bd", delta, delta)# Hadamard乘积, (batch_size, product_layer_dim)
		else:
			emb_sum = torch.sum(embedding_col, axis=1).cuda()# (batch_size, emb_dim)
			p = torch.einsum("bm, bn->bmn", emb_sum, emb_sum)
			lp = torch.einsum("bmn, dmn->bd", p, self.lp_weight)# (batch_size, product_layer_dim)
		dnn_col = torch.cat([lz, lp, unembedding_col], axis=-1)
		output = torch.relu(self.fc1(dnn_col))
		output = torch.sigmoid(self.fc2(output))
		return output

if __name__ == '__main__':
	filename = './data/criteo_sampled_data.csv'
	train_loader, test_loader, emb_col, unembed_col = get_data(filename)
    # train...
	model = PNN(emb_col, unembed_col, product_type, 10)
	model.cuda()
	total_loss = 0.0
	oof_auc = 0.0
	critetion = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
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
		auc = evaluate(model, test_loader)
		print("epoch ", i, "auc is: ", auc)
		oof_auc += auc
	print("average valid auc: ", oof_auc/epoch)
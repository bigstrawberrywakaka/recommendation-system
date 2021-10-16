"""
item based autorec
paper:
	https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf
"""
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
import random
warnings.filterwarnings('ignore')

# args
input_size = 9724
k = 64
n_class = 9724
ratio = 0.8
BATCH_SIZE = 100 
epoch = 100
N = 10000

class AutoRec(nn.Module):
	def __init__(self):
		super(AutoRec, self).__init__()
		self.input_size = input_size
		self.k = k
		self.n_class = n_class
		self.linear = nn.Sequential(
			nn.Linear(self.input_size, self.k),
			nn.Sigmoid(),
			nn.Linear(self.k, self.n_class)
			)
	def forward(self, x):
		output = self.linear(x)
		return output

def make_data(data):
	user_train_set, item_train_set, user_test_set, item_test_set = set(), set(), set(), set()
	item_len, user_len = len(list(set(data['movieId']))), len(list(set(data['userId'])))
	print("item num: ",item_len, "\nuser num: ", user_len)
	num_ratings = data.shape[0]
	random_perm_idx = np.random.permutation(num_ratings)
	train_idx = random_perm_idx[0:int(num_ratings*ratio)]
	test_idx = random_perm_idx[int(num_ratings*ratio):]
	train_user, test_user, train_mask, test_mask = np.zeros((user_len, item_len)), np.zeros((user_len, item_len)), np.zeros((user_len, item_len)), np.zeros((user_len, item_len))# 共现矩阵, mask矩阵有打分就为1，算loss
	print("train num: ", len(train_idx), "\ntest num: ", len(test_idx))
	
	# split train, test
	# 缺失值用均值填充，mask缺失值为0
	for idx in train_idx:
		user, item, rating = data.iloc[idx].values
		#if rating == 0:
		#	rating = mean(data[(data.userId == user)&(data.rating!=0)]['rating'].values)
		train_user[int(user)][int(item)] = rating
		train_mask[int(user)][int(item)] = 1
		user_train_set.add(user)
		item_train_set.add(item)
	
	for idx in test_idx:
		user, item, rating = data.iloc[idx].values
		#if rating == 0:
		#	rating = mean(data[(data.userId == user)&(data.rating!=0)]['rating'].values)
		test_user[int(user)][int(item)] = rating
		test_mask[int(user)][int(item)] = 1
		user_test_set.add(user)
		item_test_set.add(item)

	return train_user, train_mask, test_user, test_mask, user_train_set, item_train_set, user_test_set, item_test_set

def prepocess(y_pred, y, mask):
	return y_pred * mask, y * mask

def maskset(data):
	#随机mask test
	target_mask = np.zeros((len(data), len(data[0])))
	for i in range(N):# mask 1000个
		mask_userId = random.randrange(0,len(data))
		mask_itemId = random.randrange(0,len(data[0]))
		target_mask[mask_userId][mask_itemId] = 1
		data[mask_userId][mask_itemId] = 0
	return data, torch.FloatTensor(target_mask)

if __name__ == '__main__':
	# 还需要自己造数据...
	data = pd.read_csv("./data/ratings.csv")# (user, item)rating 0.5-4.5 user:610 item: 9724, 0代表没有评分
	print(data.head())
	# prepocess
	data = data[['userId', 'movieId', 'rating']]
	cat_cols = ['userId', 'movieId']
	for col in cat_cols:
	    lbl = LabelEncoder().fit(data[col])
	    data[col] = lbl.transform(data[col])
	train_user, train_mask, test_user, test_mask, user_train_set, item_train_set, user_test_set, item_test_set = make_data(data)
	train_data = TensorDataset(torch.FloatTensor(train_user), torch.FloatTensor(train_mask))
	train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
	testdata, target_mask = maskset(test_user)
	test_user = torch.FloatTensor(test_user)

	# training...
	loss_total = 0.0
	model = AutoRec()
	model.cuda()
	criterion = nn.MSELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)# l2
	for i in range(epoch):
		for idx, (user, mask) in enumerate(train_loader):
			user_batch, mask_batch = maskset(user)
			user_batch, mask_batch, user = user_batch.cuda(), mask_batch.cuda(), user.cuda()
			y_pred = model(user_batch)
			y_pred, user = prepocess(y_pred, user, mask_batch)# 后处理一下，把mask为0的值都置为0，方便计算loss
			loss = criterion(user, y_pred)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			loss_total += loss.item()
			if (idx + 1) % (len(train_loader)//5) == 0:    # 只打印五次结果
				print("Epoch {:04d} | Step {:04d}/{:04d} | RMSE Loss {:.4f}".format(i+1, idx+1, len(train_loader), sqrt(loss_total/(idx+1))))
	
	# test...
	testdata = torch.FloatTensor(testdata).cuda()
	target = model(testdata)
	target, target_mask, test_user = target.cuda(), target_mask.cuda(), test_user.cuda()
	target, test_user = prepocess(target, test_user, target_mask)
	loss = criterion(test_user, target)# 模型应该预测刚才我们手动mask的缺失值loss
	print("random mask test maxtrix 1000 RMSE Loss ", sqrt(loss))
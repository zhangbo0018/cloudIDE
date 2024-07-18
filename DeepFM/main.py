import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from model.DeepFM import DeepFM
from data.dataset import CriteoDataset

# 设置训练集的大小，900000 条数据用于训练，10000 条数据用于验证，总共 1000000 条数据
Num_train = 9000

data_content = "/cloudide/workspace/All-in-One/DeepFM/data"
# 加载训练数据
train_data = CriteoDataset(data_content, train=True)
# DataLoader 用于将数据加载为可迭代的批次，批量大小设置为 16
# 从训练数据中随机选择子集范围进行采样
loader_train = DataLoader(train_data, batch_size=16,
          sampler=sampler.SubsetRandomSampler(range(Num_train)))
print("训练集长度：",len(loader_train.sampler.indices))

# 加载验证数据
val_data = CriteoDataset(data_content, train=True)
# DataLoader 用于将数据加载为可迭代的批次，批量大小设置为 16
# 从训练数据中随机选择子集范围进行采样
loader_val = DataLoader(val_data, batch_size=16,
        sampler=sampler.SubsetRandomSampler(range(Num_train, 10000)))
print("测试集长度",len(loader_val.sampler.indices))

# 加载特征大小数据
# 从文本文件加载特征大小数据，每行数据用逗号分隔，得到一个字符串列表
feature_sizes = np.loadtxt(data_content + '/feature_sizes.txt', delimiter=',')
# 将字符串列表转换为整数列表
feature_sizes = [int(x) for x in feature_sizes]
# 打印特征大小
print("feature_size:", feature_sizes)

# 创建 DeepFM 模型
# 根据特征大小列表创建 DeepFM 模型
model = DeepFM(feature_sizes, use_cuda=False)

# 创建优化器
# 使用 Adam 优化器，学习率设置为 1e-4，权重衰减设置为 0.0
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
model.fit(loader_train, loader_val, optimizer, epochs=5, verbose=True)

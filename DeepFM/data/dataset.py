import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

# 指定连续特征的数量
continous_features = 13

# 继承 PyTorch 的 Dataset 类，创建 CriteoDataset 类
class CriteoDataset(Dataset):
    """
    用于 Criteo 数据集的自定义数据集类，以使用 PyTorch 提供的高效数据加载工具。
    """
    # 初始化方法，接收数据集的根路径和模式标识（训练或测试）
    def __init__(self, root, train=True):
        """
        初始化文件路径和训练/测试模式。
        参数:
            root: 存储处理后的数据文件的路径。
            train: 默认为 True，表示为训练数据集加载数据，当设置为 False 时，加载测试集。
        """
        self.root = root
        self.train = train
        # 检查数据文件是否存在
        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        # 如果是训练模式，加载训练数据文件并初始化实例变量
        if self.train:
            # 读取 CSV 文件
            data = pd.read_csv(os.path.join(root, 'train.txt'), header=None)
            # 获取特征数据，去除最后一列目标值
            self.train_data = data.iloc[:, :-1].values
            # 获取目标值
            self.target = data.iloc[:, -1].values
        else:
            # 如果是测试模式，加载测试数据文件并初始化实例变量
            data = pd.read_csv(os.path.join(root, 'test.txt'), header=None)
            # 获取测试数据
            self.test_data = data.iloc[:, :-1].values
    
    # 根据索引获取样本数据，返回 Xi、Xv 和 target
    """
    Xi：特征索引
    Xv：特征值
    target：目标值
    """
    def __getitem__(self, idx):
        if self.train:
            # 获取训练数据中的样本索引为 idx 的数据
            dataI, targetI = self.train_data[idx, :], self.target[idx]
            # print(dataI)
            # 创建一个与训练数据大小相同的全零 NumPy 数组，用于表示连续特征的索引
            Xi_coutinous = np.zeros_like(dataI[:continous_features])
            # 获取训练数据中的类别特征数据，索引从 continous_features 开始
            Xi_categorial = dataI[continous_features:]
            # 将连续特征索引和类别特征数据连接成一个数组，转换为 int32 类型，并使用 torch.from_numpy 创建一个张量
            Xi = torch.from_numpy(np.concatenate((Xi_coutinous, Xi_categorial)).astype(np.int32)).unsqueeze(-1)
            
            # 创建一个与类别特征数据大小相同的全一 NumPy 数组，用于表示类别特征的值（作为一热编码）
            Xv_categorial = np.ones_like(dataI[continous_features:])
            # 获取训练数据中的连续特征数据
            Xv_coutinous = dataI[:continous_features]
            # 将连续特征数据和类别特征值连接成一个数组，转换为 int32 类型，并使用 torch.from_numpy 创建一个张量
            Xv = torch.from_numpy(np.concatenate((Xv_coutinous, Xv_categorial)).astype(np.int32))
            return Xi, Xv, targetI
        else:
            # 获取测试数据中的样本索引为 idx 的数据
            dataI = self.test_data.iloc[idx, :]
            # 创建一个与测试数据大小相同的全一 NumPy 数组，用于表示连续特征的索引
            Xi_coutinous = np.ones_like(dataI[:continous_features])
            # 获取测试数据中的类别特征数据，索引从 continous_features 开始
            Xi_categorial = dataI[continous_features:]
            # 将连续特征索引和类别特征数据连接成一个数组，转换为 int32 类型，并使用 torch.from_numpy 创建一个张量
            Xi = torch.from_numpy(np.concatenate((Xi_coutinous, Xi_categorial)).astype(np.int32)).unsqueeze(-1)
            
            # 创建一个与类别特征数据大小相同的全一 NumPy 数组，用于表示类别特征的值（作为一热编码）
            Xv_categorial = np.ones_like(dataI[continous_features:])
            # 获取测试数据中的连续特征数据
            Xv_coutinous = dataI[:continous_features]
            # 将连续特征数据和类别特征值连接成一个数组，转换为 int32 类型，并使用 torch.from_numpy 创建一个张量
            Xv = torch.from_numpy(np.concatenate((Xv_coutinous, Xv_categorial)).astype(np.int32))
            return Xi, Xv

    # 返回数据集的大小，训练数据集返回训练数据的数量，测试数据集返回测试数据的数量
    def __len__(self):
        if self.train:
            # 返回训练数据的样本数量
            return len(self.train_data)
        else:
            # 返回测试数据的样本数量
            return len(self.test_data)
    # 检查指定路径下数据文件是否存在
    def _check_exists(self):
        return os.path.exists(self.root)

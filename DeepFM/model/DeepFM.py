# -*- coding: utf-8 -*-
"""
一个基于 Pytorch 的 DeepFM 实现，用于点击率预估问题。
"""
# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time
# 定义 DeepFM 网络模型
class DeepFM(nn.Module):
    """
    一个基于 Pytorch 的 DeepFM 网络，用于点击率预估问题。
    这个网络结构包含两个部分：FM 部分用于处理特征的低阶交互，Deep 部分用于处理特征的高阶交互。在这个网络中，我们对所有隐藏层使用了批量归一化（BatchNorm）和 dropout 技术，并使用 Adam 优化算法。
    你可以在这篇论文中找到更多细节：
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
    Huifeng Guo,Ruiming Tang,Yunming Ye,Zhenguo Li,Xiuqiang He.
    """
    # 初始化网络参数
    def __init__(self, feature_sizes, embedding_size=4,
                 hidden_dims=[32, 32], num_classes=1, dropout=[0.5, 0.5],
                 use_cuda=True, verbose=False):
        """
        初始化一个新的网络
        参数:
        - feature_size: 一个整型列表，给出每个字段的特征大小。
        - embedding_size: 一个整型参数， 给出特征嵌入的大小。
        - hidden_dims: 一个整型列表，给出每个隐藏层的大小。
        - num_classes: 一个整型参数，给出要预测的类别数量。例如，有人可能会对一部电影给出 1、2、3、4 或 5 颗星的评级。
        - batch_size: 一个整型参数，给出每次迭代中使用的实例数量。
        - use_cuda: 一个布尔值，指定是否使用 CUDA。
        - verbose: 一个布尔值，指定是否打印详细信息。
        """
        # 调用父类的初始化方法
        super().__init__()
        # 保存 field_size 参数
        self.field_size = len(feature_sizes)
        # 保存 feature_sizes 参数
        self.feature_sizes = feature_sizes
        # 保存 embedding_size 参数
        self.embedding_size = embedding_size
        # 保存 hidden_dims 参数
        self.hidden_dims = hidden_dims
        # 保存 num_classes 参数
        self.num_classes = num_classes
        self.dtype = torch.long
        # 使用 Parameter 类直接创建可学习参数 bias，用于偏置计算
        self.bias = torch.nn.Parameter(torch.randn(1))
        """
            检查是否使用 CUDA
        """
        # 判断是否使用 CUDA，如果可以使用且 use_cuda 为 True，则将模型放在 GPU 上
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Using GPU')
        # 否则将模型放在 CPU 上
        else:
            self.device = torch.device('cpu')
            print('Using CPU')
        """
            初始化 FM 部分
        """
        # 初始化 FM 部分的一阶嵌入层
        self.fm_first_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
        # print('fm_first_order_embeddings:', self.fm_first_order_embeddings)
        total_params = 0
        for embedding_layer in self.fm_first_order_embeddings:
            total_params += sum(p.numel() for p in embedding_layer.parameters())
        print("fm_first_order_embeddings层的权重参数数量：", total_params)

        # 初始化 FM 部分的二阶嵌入层
        self.fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
        # print('fm_second_order_embeddings:', self.fm_second_order_embeddings)
        total_params = 0
        for embedding_layer in self.fm_second_order_embeddings:
            total_params += sum(p.numel() for p in embedding_layer.parameters())
        print("fm_second_order_embeddings层的权重参数数量：", total_params)
        """
            初始化 Deep 部分
        """
        # 计算所有层的大小，包括输入层、隐藏层和输出层
        all_dims = [self.field_size * self.embedding_size] + \
            self.hidden_dims + [self.num_classes]
        # 使用序列操作初始化线性层，批量归一化和 dropout 层
        for i in range(1, len(hidden_dims) + 1):
            # 初始化线性层
            setattr(self, 'linear_' + str(i),nn.Linear(all_dims[i - 1], all_dims[i]))
            # 使用 kaiming_normal_方法初始化 fc1 的权重
            # nn.init.kaiming_normal_(self.fc1.weight)
            # 初始化批量归一化层
            setattr(self, 'batchNorm_' + str(i),nn.BatchNorm1d(all_dims[i]))
            # 初始化 dropout 层
            setattr(self, 'dropout_' + str(i),nn.Dropout(dropout[i - 1]))
            #  # 打印当前循环次数
            # print("第", i, "次循环：")
            # # 打印创建的线性层对象
            # print("创建的线性层对象:", getattr(self, 'linear_' + str(i)))
            # # 打印创建的批量归一化层对象
            # print("创建的批量归一化层对象:", getattr(self, 'batchNorm_' + str(i)))
            # # 打印创建的 dropout 层对象
            # print("创建的 dropout 层对象:", getattr(self, 'dropout_' + str(i)))
    # 定义前向传播函数
    def forward(self, Xi, Xv):
        """
        网络的前向传播过程。
        参数:
        - Xi: 输入的索引张量，形状为 (N, field_size, 1)。
        - Xv: 输入的值张量，形状为 (N, field_size, 1)。
        """
        """
            FM 部分
        """
        # # 遍历 self.fm_first_order_embeddings 列表
        # for i,emb in enumerate(self.fm_first_order_embeddings):
        #     # 打印 Xi 矩阵的第 i 列的形状
        #     print(Xi[:,i,:].shape)
        #     # 打印嵌入层 emb 对 Xi 矩阵的第 i 列的操作结果的形状
        #     print(emb(Xi[:,i,:]).shape)
        #     # 打印 Xv 矩阵的第 i 列的形状
        #     print(Xv[:,i].shape)
        #     # 打印 Xv 矩阵的第 i 列的转置的形状
        #     print(Xv[:,i].t().shape)
        # 计算 FM 部分的一阶嵌入结果
        fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                  enumerate(self.fm_first_order_embeddings)]
        # 将 FM 一阶嵌入结果拼接成一个矩阵
        fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
        # 打印 fm_first_order 的形状
        # print(fm_first_order.shape)
        # 计算 FM 部分的二阶嵌入结果
        fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                   enumerate(self.fm_second_order_embeddings)]
        # 将 FM 二阶嵌入结果相加，得到二阶嵌入结果总和
        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        # 计算二阶嵌入总和的平方
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * \
                                         fm_sum_second_order_emb  # (x+y)^2
        # 打印 fm_sum_second_order_emb_square 的形状
        # print(fm_sum_second_order_emb_square.shape)
        # 计算每个二阶嵌入结果的平方
        fm_second_order_emb_square = [
            item * item for item in fm_second_order_emb_arr]
        # 将所有二阶嵌入结果的平方相加，得到平方总和
        fm_second_order_emb_square_sum = sum(
            fm_second_order_emb_square)  # x^2+y^2
        # 计算 FM 部分的二阶结果
        fm_second_order = (fm_sum_second_order_emb_square -
                           fm_second_order_emb_square_sum) * 0.5
        """
            Deep 部分
        """
        # 将 FM 二阶嵌入结果拼接，作为深度部分的输入
        deep_emb = torch.cat(fm_second_order_emb_arr, 1)
        # 将拼接后的 FM 二阶嵌入结果作为深度部分第一个全连接层的输入
        deep_out = deep_emb
        # 对深度部分的每一层进行前向传播
        for i in range(1, len(self.hidden_dims) + 1):
            # 从模型属性中获取当前层的线性变换操作
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            # 从模型属性中获取当前层的批量归一化操作
            deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            # 从模型属性中获取当前层的辍学操作
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)
        """
            总和
        """
        # 将 FM 的一阶结果、二阶结果和深度部分的结果相加，并加上偏置项
        total_sum = torch.sum(fm_first_order, 1) + \
                    torch.sum(fm_second_order, 1) + torch.sum(deep_out, 1) + self.bias
        return total_sum
    # 定义拟合函数，用于训练和评估模型
    def fit(self, loader_train, loader_val, optimizer, epochs=100, verbose=False, print_every=100):
        """
        训练模型并验证准确性。
        参数:
        - loader_train: 训练集数据加载器。
        - loader_val: 验证集数据加载器。
        - optimizer: 训练过程中使用的优化器抽象，例如，“torch.optim.Adam()”“torch.optim.SGD()”。
        - epochs: 迭代次数。
        - verbose: 打印信息的布尔值。
        - print_every: 打印信息的间隔，即每多少次迭代打印一次。
        """
        """
            加载输入数据
        """
        # 将模型设置为训练模式，并移动到指定的设备上（GPU 或 CPU）
        model = self.train().to(device=self.device)
        # 定义损失函数，这里使用二元交叉熵损失函数
        criterion = F.binary_cross_entropy_with_logits
        for _ in range(epochs):
            for t, (xi, xv, y) in enumerate(loader_train):
                # 将输入数据移动到与模型相同的设备上，并进行相应的数据类型转换
                xi = xi.to(device=self.device, dtype=self.dtype)
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.float)
                # 执行前向传播，计算模型的输出
                total = model(xi, xv)
                # 使用损失函数计算输出与真实值之间的损失
                loss = criterion(total, y)
                # 将优化器中的梯度置为零，以便进行下一次反向传播
                optimizer.zero_grad()
                # 执行反向传播，计算损失相对于模型参数的梯度
                loss.backward()
                # 调用优化器的step方法，根据梯度更新模型的参数
                optimizer.step()

                if verbose and t % print_every == 0:
                    # 打印迭代次数和当前的损失值
                    print('Iteration %d, loss = %.4f' % (t, loss.item()))
                    # 在验证集上检查模型的准确性
                    self.check_accuracy(loader_val, model)
                    print()
    
    # 定义检查准确性的函数，用于评估模型在验证集或测试集上的性能
    def check_accuracy(self, loader, model):
        # 判断当前数据加载器中的数据集用途是训练集还是测试集
        if loader.dataset.train:
            # 如果是训练集，打印信息是验证集的准确性
            print('Checking accuracy on validation set')
        else:
            # 如果是测试集，打印信息是测试集的准确性
            print('Checking accuracy on test set')   
        # 初始化正确预测的样本数和总样本数
        num_correct = 0
        num_samples = 0
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            # 遍历数据加载器中的每个批次
            for xi, xv, y in loader:
                # 将输入数据移动到与模型相同的设备上，并进行相应的数据类型转换
                xi = xi.to(device=self.device, dtype=self.dtype)
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.bool)
                # 执行前向传播，计算模型的输出
                total = model(xi, xv)
                # 将输出转换为sigmoid函数作用后的概率，并与0.5进行比较得到预测结果
                preds = (F.sigmoid(total) > 0.5)
                # 统计预测正确的样本数
                num_correct += (preds == y).sum()
                # 更新总样本数
                num_samples += preds.size(0)
            # 计算准确率
            acc = float(num_correct) / num_samples
            # 打印正确预测的样本数、总样本数和准确率
            print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))               

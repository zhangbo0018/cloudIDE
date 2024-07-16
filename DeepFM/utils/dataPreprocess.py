
"""
预处理Criteo数据集。该数据集用于显示广告挑战（https://www.kaggle.com/c/criteo-display-ad-challenge）。
"""

import os
import sys
import click
import random
import collections

# 有13个整数特征和26个分类特征
continous_features = range(1, 14)
categorial_features = range(14, 40)

# 裁剪整数特征。每个整数特征的裁剪点来自于每个特征中总价值的95%分位数
continous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]


class CategoryDictGenerator:
    """
    为每个分类特征生成字典
    """

    def __init__(self, num_feature):
        """
        初始化函数

        参数：
            num_feature (int): 特征数量

        返回：
            无

        说明：
            初始化时，创建指定数量的字典，并将每个字典初始化为一个空的 defaultdict(int)
        """
        self.dicts = []
        # 设置属性 num_feature
        self.num_feature = num_feature
        # 初始化 collections.defaultdict(int) 列表
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    def build(self, datafile, categorial_features, cutoff=0):
        """
        构建字典函数

        参数：
            datafile (str): 数据文件路径
            categorial_features (list): 类别特征列表
            cutoff (int): 字典中条目的最小出现次数

        返回：
            无

        说明：
            构建每个特征的字典，包含满足最小出现次数的条目
        """

        # 打开数据文件，以只读模式
        with open(datafile, 'r') as f:
            # 遍历数据文件的每一行
            for line in f:
                # 移除行末的换行符并将行拆分为特征列表
                features = line.rstrip('\n').split('\t')
                # 遍历每个特征
                for i in range(0, self.num_feature):
                    # 如果特征值不为空
                    if features[categorial_features[i]]!= '':
                        # 增加相应特征字典中的计数
                        self.dicts[i][features[categorial_features[i]]] += 1
        # 遍历每个特征字典
        for i in range(0, self.num_feature):
            # 过滤掉字典中计数小于截止值的项
            self.dicts[i] = filter(lambda x: x[1] >= cutoff,
                                   self.dicts[i].items())
            # 根据计数对字典中的项进行排序，先按计数降序，再按键升序
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            # 获取排序后的键值对并解压为vocabs和其他
            vocabs, _ = list(zip(*self.dicts[i]))
            # 将vocabs转换为字典，并为每个唯一的vocab分配一个从1开始的整数ID
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            # 为每个字典添加一个特殊的'<unk>'键，并将其值设置为0
            self.dicts[i]['<unk>'] = 0

    def gen(self, idx, key):
        """
        生成对应索引和键的值

        参数：
            idx (int): 字典索引
            key (str): 要查找的值

        返回：
            int: 如果键存在于字典中，则返回其值；否则，返回代表未知的数字

        说明：
            如果键不存在于字典中，返回 0，代表未知的数字
        """
        # 如果键不在字典中，则获取字典中键为<unk>的值
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        # 否则获取字典中当前键对应的值
        else:
            res = self.dicts[idx][key]
        # 返回结果
        return res

    def dicts_sizes(self):
        """
        获取字典大小

        参数：
            无

        返回：
            list: 包含每个字典大小的列表

        说明：
            返回一个列表，其中每个元素是对应的字典的大小
        """
        return [len(self.dicts[idx]) for idx in range(0, self.num_feature)]


class ContinuousFeatureGenerator:
    """
    裁剪连续特征。
    """

    def __init__(self, num_feature):
        self.num_feature = num_feature

    def build(self, datafile, continous_features):
        # 打开数据文件，以只读模式
        with open(datafile, 'r') as f:
            # 读取文件中的每一行数据
            for line in f:
                # 去除每行数据结尾的换行符，并按制表符分割成特征列表
                features = line.rstrip('\n').split('\t')
                # 遍历所有连续型特征
                for i in range(0, self.num_feature):
                    # 获取当前连续型特征的值
                    val = features[continous_features[i]]
                    # 如果值存在且不为空
                    if val!= '':
                        # 将值转换为整数
                        val = int(val)
                        # 如果值大于当前连续型特征的最大值
                        if val > continous_clip[i]:
                            # 将值设置为当前连续型特征的最大值
                            val = continous_clip[i]

    def gen(self, idx, val):
        # 如果值为空字符串，则返回 0.0
        if val == '':
            return 0.0
        # 将值转换为浮点数
        val = float(val)
        # 返回转换后的值
        return val


# @click.command("preprocess")
# @click.option("--datadir", type=str, help="Path to raw criteo dataset")
# @click.option("--outdir", type=str, help="Path to save the processed data")
def preprocess(datadir, outdir, num_train_sample = 10000, num_test_sample = 10000):
    """
    所有13个整数特征被归一化为连续值，这些连续特征被合并为一个维度为13的向量。
    26个分类特征中的每一个都进行了one-hot编码，所有的one-hot向量被合并为一个稀疏的二进制向量。
    """

    # 初始化连续特征生成器，参数为连续特征的数量
    dists = ContinuousFeatureGenerator(len(continous_features))
    # 构建特征字典，参数为数据文件路径和连续特征列表
    dists.build(os.path.join(datadir, 'train.txt'), continous_features)

    # 初始化类别字典生成器，参数为类别特征的数量
    dicts = CategoryDictGenerator(len(categorial_features))
    # 构建类别字典，参数为数据文件路径、类别特征列表和截断值
    dicts.build(os.path.join(datadir, 'train.txt'), categorial_features, cutoff=200)

    # 获取类别字典的大小
    dict_sizes = dicts.dicts_sizes()
    # 打开输出路径下的文件，若没有该文件则创建新文件，并将文件句柄保存到变量 feature_sizes 中
    with open(os.path.join(outdir, 'feature_sizes.txt'), 'w') as feature_sizes:
        # 初始化一个列表，其长度等于连续特征的数量加上类别字典的大小
        sizes = [1] * len(continous_features) + dict_sizes
        # 将列表中的每个元素转换为字符串
        sizes = [str(i) for i in sizes]
        # 将列表元素用逗号分隔，连接成一个字符串
        feature_sizes.write(','.join(sizes))

    # 设置随机数种子，确保每次运行的随机性是一致的，方便代码复现
    random.seed(0)

    # 保存用于训练的数据。

    # 打开位于 outdir 目录下的 train.txt 文件，准备写入数据
    with open(os.path.join(outdir, 'train.txt'), 'w') as out_train:
        # 同时，打开位于 datadir 目录下的 train.txt 文件，准备读取数据
        with open(os.path.join(datadir, 'train.txt'), 'r') as f:
            # 初始化计数器 k，用于记录已处理的行数
            k = 0
            # 遍历文件 f 中的每一行
            for line in f:
                # 移除行尾的换行符，并拆分为特征列表
                features = line.rstrip('\n').split('\t')

                # 初始化连续特征值列表
                continous_vals = []
                # 遍历所有连续特征
                for i in range(0, len(continous_features)):
                    # 根据特征索引和当前行的数据，生成连续特征值
                    val = dists.gen(i, features[continous_features[i]])
                    # 将连续特征值格式化为小数点后 6 位，并进行处理
                    continous_vals.append("{0:.6f}".format(val).rstrip('0').rstrip('.'))

                # 初始化类别特征值列表
                categorial_vals = []
                # 遍历所有类别特征
                for i in range(0, len(categorial_features)):
                    # 根据特征索引和当前行的数据，生成类别特征值
                    val = dicts.gen(i, features[categorial_features[i]])
                    # 将类别特征值转换为字符串
                    categorial_vals.append(str(val))

                # 将连续特征值列表转换为逗号分隔的字符串
                continous_vals = ','.join(continous_vals)
                # 将类别特征值列表转换为逗号分隔的字符串
                categorial_vals = ','.join(categorial_vals)
                # 获取当前行的标签值
                label = features[0]
                # 将连续特征值、类别特征值和标签值连接为一个字符串，并写入到 out_train 文件中
                out_train.write(','.join([continous_vals, categorial_vals, label]) + '\n')

                # 更新计数器 k
                k += 1
                # 如果处理的行数达到或超过指定的 num_train_sample，则停止循环
                if k >= num_train_sample:
                    break

    with open(os.path.join(outdir, 'test.txt'), 'w') as out:
        with open(os.path.join(datadir, 'test.txt'), 'r') as f:
            k = 0
            for line in f:
                features = line.rstrip('\n').split('\t')

                continous_vals = []
                for i in range(0, len(continous_features)):
                    val = dists.gen(i, features[continous_features[i] - 1])
                    continous_vals.append("{0:.6f}".format(val).rstrip('0')
                                         .rstrip('.'))
                categorial_vals = []
                for i in range(0, len(categorial_features)):
                    val = dicts.gen(i, features[categorial_features[
                        i] - 1])
                    categorial_vals.append(str(val))

                continous_vals = ','.join(continous_vals)
                categorial_vals = ','.join(categorial_vals)
                out.write(','.join([continous_vals, categorial_vals]) + '\n')
                k += 1
                if k >= num_test_sample:
                    break

if __name__ == "__main__":
    preprocess('../data/raw', '../data')
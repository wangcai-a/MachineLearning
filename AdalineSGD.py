import numpy as np
from numpy.random import seed


class AdalineSGD(object):
    '''
    自适应神经元分类器

    Paramenter(超参)
    -----------
    eta: float
        学习率(0到1之间)
    n_iter: int
        迭代次数

    Attributs
    ----------
    w_: 1d_array
        训练后的权重
    errors_: list
        每次训练的错误数
    shuffle: bool(default: True)
        shuffle方法用来重排训练数据
    random_state: int(default:None)
        随机数种子
        设置随机数种子进行重排,并初始化权重
    '''
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_satte=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_satte:
            seed(random_satte)  # 使用随机数种子后,每次随机出的数相同

    def fit(self, x, y):
        '''
        训练数据函数
        :param x(DataDrame对象): {array-like}, shape=[n_samples, n_features]
                  训练向量(训练集)
                  n_samples: 样本数量
                  n_features: 特征数量
        :param y: array-like, shape=[n_smaples]
                  目标值(结果集)
        :param w_: 权重向量
        :return:
        '''
        self._initialize_weights(x.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                x, y = self._shuffle(x, y)
            cost = []
            for xi, target in zip(x, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, x, y):
        '''
        训练部分数据
        :param x:
        :param y:
        :return:
        '''
        if not self.w_initialized:
            self._initialize_weights(x.shape[1])
        if y.ravel().shape[0] > 1:  # np.revel()方法将多维数组降一维
            for xi, target in zip(x, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(x, y)
        return self

    def _shuffle(self, x, y):
        '''
        将数据进行重排
        :param x:
        :param y:
        :return:
        '''
        # 传入数组的长度,产生一个随机序列作为索引/若传入数据集则调用random.shuffle方法生成随机数据集
        r = np.random.permutation(len(y))
        return x[r], y[r]   # 使用新的索引从原数据集生成新的随机数据集

    def _initialize_weights(self, m):
        '''
        初始化权重
        :param m:
            训练数组的长度
        :return:
        '''
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        '''
        使用Adaline规则更新权重
        :param xi:
        :param target:
        :return:
        '''
        out_put = self.net_input(xi)
        error = (target - out_put)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, x):
        '''
        计算净输入
        对输入向量和权重向量做点积
        :param x:
        :return:
            两个矩阵的内积
        '''
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def activation(self, x):
        '''
        线性激活函数
        :param x:
        :return:
        '''
        return self.net_input(x)

    def predict(self, x):
        '''
        在下次数据测试数据后输出预测标签
        :param x:
        :return:
            大于等于0输出1,
            否则输出-1
        '''
        return np.where(self.activation(x) >= 0.0, 1, -1)
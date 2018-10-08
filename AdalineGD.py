import numpy as np


class AdalineGD(object):
    '''
    自适应神经元分类器

    Paramenter
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
    '''
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y):
        '''
        训练数据函数
        :param x: {array-like}, shape=[n_samples, n_features]
                  训练向量(训练集)
                  n_samples: 样本数量
                  n_features: 特征数量
        :param y: array-like, shape=[n_smaples]
                  目标值(结果集)
        :return:
        '''
        self.w_ = np.zeros(1 + x.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            output = self.net_input(x)
            errors = (y - output)
            self.w_[1:] += self.eta * x.T.dot(errors)
            self.w_[0:] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, x):
        '''
        计算净输入
        :param x:
        :return:
            两个矩阵的内积
            注:x.T.dot(y)等于np.dot(x.T, y)
        '''
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def activation(self, x):
        '''
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
        return np.where(self.net_input(x) >= 0.0, 1, -1)
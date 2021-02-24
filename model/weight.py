import numpy as np
from model.exception import WeightsInitializeError


def weight_initializer(dim, train_bias=True, mode='zero'):
    """
    权重初始化
    :param dim: 参数的个数
    :param train_bias: 是否训练偏执量
    :param mode: 初始化类型
    :return: 权重矩阵
    """
    if mode == 'zero':
        bias = np.array([[0]])
        weights = np.zeros((dim, 1))
    elif mode == 'normal':
        bias = np.array([[0]])
        weights = np.random.rand(dim, 1)
    else:
        raise WeightsInitializeError("Unknown flags.")

    return np.concatenate((weights, bias)) if train_bias == True else weights


if __name__ == '__main__':
    w = weight_initializer(3, mode='normal', train_bias=False)
    print(w)

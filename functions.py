import numpy as np


#活性化関数
def identity(x):
    return x


def step(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x = x - np.max(x,axis=-1,keepdims=True)
    exp = np.exp(x)
    sum_exp = np.sum(exp,axis=-1,keepdims=True)
    return exp / sum_exp


#損失関数
def mean_squared_error(y, t):
    n = y.shape[0]
    return 0.5 * np.sum((y - t)**2) / n


def cros_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    n = y.shape[0]

    #正解ラベルがone-hot行列のとき
    if t.size == y.size:
        #return -np.sum(t*np.log(y+1e-7))/n
        t = t.argmax(axis=1)

    return -np.sum(np.log(y[np.arange(n), t] + 1e-7)) / n

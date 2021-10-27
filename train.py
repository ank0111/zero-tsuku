import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

#ハイパーパラメータ
iter_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(784,50,10)

for i in range(iter_num):
    #ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #勾配の計算
    grad = network.gradient(x_batch, t_batch)

    #パラメータの更新
    for key in ('W1','b1','W2','b2'):
        network.params[key] -= learning_rate * grad[key]

    #学習経過の記録
    if i%1000 == 0:
        print(network.accuracy(x_train,t_train),network.accuracy(x_test,t_test))
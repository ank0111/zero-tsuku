import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)

network = TwoLayerNet(10, 10,10)

x_batch = x_train[:3,:10]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np._average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))
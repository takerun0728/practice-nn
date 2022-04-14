import matplotlib.pyplot as plt
import numpy as np
from neural_lib import *
np.random.seed(0)

DEF_MID_NEURON_NUM = 32
DEF_OPT = 'momentum'
DEF_OPT_PARAM = {'eta':0.01, 'rho':0.9, 'alpha':0.9}
DEF_LAMBDA = 0.0
EPOCH = 10000
LOG_INTERVAL = 1
DRAW_INTERVAL = 1000
BATCH = 16
N_TIME = 8

class MultOutRNNNetwork(AbstractNetwork):
    def __init__(self, mid_neuron_num=DEF_MID_NEURON_NUM, optimizer=DEF_OPT, opt_params=DEF_OPT_PARAM, lam=DEF_LAMBDA):
        super().__init__()
        self.layers = [RNNLayer(2, mid_neuron_num, optimizer=optimizer, opt_params=opt_params, lam=lam, is_out_mult=True)]
        self.layers.append(Layer(mid_neuron_num, 1, activate_func='sigmoid', opt_params=opt_params, lam=lam))

    def forward(self, x):
        rnn_out = self.layers[0].forward(x, self.is_train)
        n_b, n_t, n_o = rnn_out.shape
        rnn_out = rnn_out.reshape(n_b * n_t, n_o)
        tmp = self.layers[1].forward(rnn_out, self.is_train)
        return tmp.reshape(n_b, n_t, -1)

    def backward(self, y_label):
        n_b, n_t, n_y = y_label.shape
        y_label = y_label.reshape(n_b * n_t, n_y)
        grad = self.layers[1].backward(y_label)
        grad = self.layers[0].backward(grad.reshape(n_b, n_t, -1))

if __name__ == '__main__':
    network = MultOutRNNNetwork()

    max_num = 2**N_TIME
    binaries = np.zeros((max_num, N_TIME), dtype=np.int32)
    for i in range(max_num):
        num10 = i
        for j in range(N_TIME):
            binaries[i, j] = num10 % 2
            num10 //= 2

    for i in range(EPOCH): #This EPOCH is uncorrect
        network.set_train_mode(True)

        num1 = np.random.randint(0, max_num // 2, size=BATCH)
        num2 = np.random.randint(0, max_num // 2, size=BATCH)
        ans = num1 + num2
        x = np.zeros((BATCH, N_TIME, 2))
        x[:, :, 0] = binaries[num1]
        x[:, :, 1] = binaries[num2]
        y_label = np.zeros((BATCH, N_TIME, 1))
        y_label[:, :, 0] = binaries[ans]

        y_inference = network.forward(x)
        network.backward(y_inference - y_label)
        network.update()

        if i % LOG_INTERVAL == 0:
            error = np.sum(np.square(y_inference - y_label))
            print(f'Epoch: {i}/{EPOCH}')
            print(f'Error: {error / BATCH}')



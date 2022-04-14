import matplotlib.pyplot as plt
import numpy as np
from neural_lib import *

DEF_MID_NEURON_NUM = 20
DEF_OPT = 'momentum'
DEF_OPT_PARAM = {'eta':0.01, 'rho':0.9, 'alpha':0.9}
DEF_LAMBDA = 0.001
EPOCH = 10000
LOG_INTERVAL = 100
DRAW_INTERVAL = 1000
BATCH = 8
N_TIME = 10
NOISE_AMP = 0.1

class RNNNetwork(AbstractNetwork):
    def __init__(self, mid_neuron_num=DEF_MID_NEURON_NUM, optimizer='momentum', opt_params=DEF_OPT_PARAM, lam=DEF_LAMBDA):
        super().__init__()
        self.layers = [RNNLayer(1, mid_neuron_num, activate_func='sigmoid',optimizer=optimizer, opt_params=opt_params, lam=lam)]
        self.layers.append(Layer(mid_neuron_num, 1, activate_func='identify', opt_params=opt_params, lam=lam))

if __name__ == '__main__':
    network = RNNNetwork()

    sin_x = np.linspace(-4*np.pi, 4*np.pi)
    sin_y = np.sin(sin_x) + NOISE_AMP * np.random.randn(len(sin_x))
    n_sample = len(sin_x) - N_TIME
    input_data = np.zeros((n_sample, N_TIME, 1))
    correct_data = np.zeros((n_sample, 1))

    for i in range(0, n_sample):
        input_data[i, :, 0] = sin_y[i:i+N_TIME]
        correct_data[i, 0] = sin_y[i+N_TIME]

    indexes = np.arange(n_sample)
    y_epoch = []

    for i in range(EPOCH):
        network.set_train_mode(True)
        np.random.shuffle(indexes)
        
        error = 0
        start = 0
        plot_x = []
        plot_y = []
        log_flag = False
        draw_flag = False

        if i % LOG_INTERVAL == 0:
            log_flag = True

        if i % DRAW_INTERVAL == 0:
            draw_flag = True

        while start < n_sample:
            x = input_data[indexes[start:start+BATCH]]
            y_label = correct_data[indexes[start:start+BATCH]]
            
            y_inference = network.forward(x)    
            network.backward(y_inference - y_label)
            network.update()
                
            if log_flag:
                error += np.sum(np.square(y_inference - y_label))

            start += BATCH

        if log_flag:
            print(f'Epoch: {i}/{EPOCH}')
            print(f'Error: {error / n_sample}')

        network.set_train_mode(False)

        if draw_flag:
            predicted = input_data[0].reshape(-1).tolist()
            for i in range(n_sample):
                x = np.array(predicted[-N_TIME:]).reshape(1, N_TIME, 1)
                y = network.forward(x)
                predicted.append(y[0, 0])

            plt.plot(sin_y)
            plt.plot(predicted)
            plt.show()
            pass


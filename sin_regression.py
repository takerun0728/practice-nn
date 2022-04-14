import matplotlib.pyplot as plt
import numpy as np
from neural_lib import *

MID_LAYER_NUM = 3
MID_NEURON_NUM = 20
EPOCH = 1000001
LOG_INTERVAL = 1000
DRAW_INTERVAL = 10000
INPUT_DATA = np.arange(-1, 1, 0.01)
OUTPUT_DATA = np.sin(INPUT_DATA * np.pi)
LEARN_DATA = OUTPUT_DATA + np.sin(INPUT_DATA * np.pi * 5) * 0.3
N_DATA = len(OUTPUT_DATA)
BATCH = 64

class RegressionNetwork(AbstractNetwork):
    def __init__(self, mid_layer_num, mid_neuron_num, optimizer='rmsprop', opt_params={'eta':0.001, 'rho':0.9, 'alpha':0.9}, lam=0.1):
        self.layers = [Layer(1, mid_neuron_num, activate_func='sigmoid', wb_width=1, optimizer=optimizer, opt_params=opt_params, lam=lam)]
        for _ in range(mid_layer_num - 1):
            self.layers.append(Layer(mid_neuron_num, mid_neuron_num, activate_func='sigmoid',  wb_width=1, optimizer=optimizer, opt_params=opt_params, lam=lam))
        self.layers.append(OutputLayer(mid_neuron_num, 1, activate_func='identify', opt_params=opt_params, lam=lam))

if __name__ == '__main__':
    network = RegressionNetwork(MID_LAYER_NUM, MID_NEURON_NUM)
    indexes = np.arange(N_DATA)
    y_epoch = []

    for i in range(EPOCH):
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

        while start < N_DATA:
            x = INPUT_DATA[indexes[start:start+BATCH]]
            y_label = LEARN_DATA[indexes[start:start+BATCH]]
            y_inference = network.forward(x.reshape(-1, 1))
            network.backward(y_label.reshape(-1, 1))
            network.update()
            
            if log_flag:
                y_inference = y_inference.reshape(-1)
                error += np.sum(np.square(y_inference - y_label))

            if draw_flag:
                y_inference = y_inference.reshape(-1)
                plot_x.extend(x)
                plot_y.extend(y_inference)

            start += BATCH

        if log_flag:
            print(f'Epoch: {i}/{EPOCH}')
            print(f'Error: {error / N_DATA}')

        if draw_flag:
            plt.plot(INPUT_DATA, OUTPUT_DATA, ls='dashed', color='red')
            plt.plot(INPUT_DATA, LEARN_DATA, ls='dashed', color='orange')
            plt.scatter(plot_x, plot_y, marker='+')
            plt.show()

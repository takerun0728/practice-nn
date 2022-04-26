import matplotlib.pyplot as plt

GPU = False
if GPU:
    import cupy as np
else:
    import numpy as np

from neural_lib import *

MID_LAYER_NUM = 3
MID_NEURON_NUM = 6
EPOCH = 100001
LOG_INTERVAL = 1000
DRAW_INTERVAL = 10000
INPUT_GAP = 0.1
BATCH = 64
EPS = 1e-7

class ClassificationNetwork(AbstractNetwork):
    def __init__(self, mid_layer_num, mid_neuron_num, activate_func='relu', optimizer='sgd', opt_params={'eta':0.001, 'rho':0.9, 'alpha':0.9}, lam=0.01):
        super().__init__()
        self.layers = [Layer(2, mid_neuron_num, activate_func=activate_func, optimizer=optimizer, opt_params=opt_params, lam=lam)]
        for _ in range(mid_layer_num - 1):
            self.layers.append(Layer(mid_neuron_num, mid_neuron_num, activate_func=activate_func,  optimizer=optimizer, opt_params=opt_params, lam=lam))
        self.layers.append(OutputLayer(mid_neuron_num, 2, activate_func='softmax', opt_params=opt_params, lam=lam))

def generate_train_data():
    x = np.arange(-1, 1, INPUT_GAP)
    y = np.arange(-1, 1, INPUT_GAP)
    y_sin = np.sin(x * np.pi)
    xx, yy = np.meshgrid(x, y)
    if GPU:
        x = x.get()
        y_sin = y_sin.get()
    xx = xx.flatten()
    yy = yy.flatten()
    data_in = np.stack([xx, yy], 1)
    data_out0 = np.where(yy < np.sin(xx * np.pi), 0, 1)
    data_out1 = np.where(yy < np.sin(xx * np.pi), 1, 0)
    return x, y_sin, data_in, np.stack([data_out0, data_out1], 1)

if __name__ == '__main__':
    x, y_sin, data_in, data_out = generate_train_data()
    network = ClassificationNetwork(MID_LAYER_NUM, MID_NEURON_NUM)
    indexes = np.arange(len(data_out))

    for i in range(EPOCH):
        np.random.shuffle(indexes)
        
        error = 0
        start = 0
        plot_x1 = []
        plot_y1 = []
        plot_x2 = []
        plot_y2 = []
        log_flag = False
        draw_flag = False

        if i % LOG_INTERVAL == 0:
            log_flag = True

        if i % DRAW_INTERVAL == 0:
            draw_flag = True

        while start < len(data_out):
            d_in = data_in[indexes[start:start+BATCH], :]
            out_label = data_out[indexes[start:start+BATCH], :]
            inference = network.forward(d_in)
            network.backward(out_label)
            network.update()
            
            if log_flag:
                error += -np.sum(out_label * np.log(inference + EPS))

            if draw_flag:
                if GPU:
                    tmp_d_in = d_in.get()
                else:
                    tmp_d_in = d_in
                plot_x1.extend([x for x, out in zip(tmp_d_in[:, 0], inference) if out[0] > out[1]])
                plot_y1.extend([y for y, out in zip(tmp_d_in[:, 1], inference) if out[0] > out[1]])
                plot_x2.extend([x for x, out in zip(tmp_d_in[:, 0], inference) if out[0] <= out[1]])
                plot_y2.extend([y for y, out in zip(tmp_d_in[:, 1], inference) if out[0] <= out[1]])

            start += BATCH

        if log_flag:
            print(f'Epoch: {i}/{EPOCH}')
            print(f'Error: {error / len(data_out)}')

        if draw_flag:
            plt.plot(x, y_sin, ls='dashed')
            plt.scatter(plot_x1, plot_y1, marker='+')
            plt.scatter(plot_x2, plot_y2, marker='o')
            plt.show()

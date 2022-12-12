from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from neural_lib import *

MID_NEURON_NUM = 32
EPOCH = 4000
LOG_INTERVAL = 10
DRAW_INTERVAL = 1000
BATCH = 32

class AutoEncoder(AbstractNetwork):
    def __init__(self, mid_neuron_num=MID_NEURON_NUM, activate_func='relu', optimizer='momentum', opt_params={'eta':0.001, 'rho':0.9, 'alpha':0.9}, lam=0.0, dropout=0.0):
        super().__init__()
        self.layers = [Layer(64, mid_neuron_num, activate_func=activate_func, optimizer=optimizer, opt_params=opt_params, lam=lam)]
        self.layers.append(Layer(mid_neuron_num, mid_neuron_num, activate_func=activate_func, optimizer=optimizer, opt_params=opt_params, lam=lam))
        self.layers.append(Layer(mid_neuron_num, 64, activate_func='sigmoid', opt_params=opt_params, lam=lam))

if __name__ == '__main__':
    network = AutoEncoder()
    didits = datasets.load_digits()
    input_data = didits.data / 15
    n_data = len(input_data)
    idxes = np.arange(n_data)

    error_train_list = []
    for i in range(EPOCH):
        np.random.shuffle(idxes)
        
        error_train = 0
        start = 0
        log_flag = False

        if i % LOG_INTERVAL == 0:
            log_flag = True

        network.set_train_mode(True)
        while start < n_data:
            input = input_data[idxes[start:start+BATCH], :]
            inference = network.forward(input)
            network.backward(inference - input)
            network.update()
            
            if log_flag:
                error_train += np.sum(np.square(inference - input))
            start += BATCH

        network.set_train_mode(False)
        if log_flag:
            error_train /= n_data

            print(f'Epoch: {i}/{EPOCH}')
            print(f'Error Train: {error_train}')

        if i % DRAW_INTERVAL == 0:
            fig = plt.figure()
            for j in range(10):
                inference = network.forward(input_data[j])
                ax = fig.add_subplot(2, 10, j + 1)
                ax.imshow(input_data[j].reshape(8, 8))
                ax = fig.add_subplot(2, 10, j + 11)
                ax.imshow(inference.reshape(8, 8))
            plt.show()



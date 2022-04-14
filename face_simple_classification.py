from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from neural_lib import *

MID_NEURON_NUM = 24
EPOCH = 4000
LOG_INTERVAL = 10
BATCH = 8

class FaceClassificationNetwork(AbstractNetwork):
    def __init__(self, mid_neuron_num, activate_func='relu', wb_width=0.1, optimizer='momentum', opt_params={'eta':0.001, 'rho':0.9, 'alpha':0.9}, lam=0.2, dropout=0.0):
        super().__init__()
        self.layers = [Layer(4096, mid_neuron_num, activate_func=activate_func, wb_width=wb_width, optimizer=optimizer, opt_params=opt_params, lam=lam)]
        self.layers.append(Dropout(dropout))
        self.layers.append(Layer(mid_neuron_num, mid_neuron_num, activate_func=activate_func,  wb_width=wb_width, optimizer=optimizer, opt_params=opt_params, lam=lam))
        self.layers.append(Dropout(dropout))
        self.layers.append(OutputLayer(mid_neuron_num, 40, activate_func='softmax', opt_params=opt_params, lam=lam))

if __name__ == '__main__':
    network = FaceClassificationNetwork(MID_NEURON_NUM)
    face_data = datasets.fetch_olivetti_faces()
    input_data = face_data.data
    correct = face_data.target
    n_data = len(correct)

    ave_input = np.average(input_data, axis=0)
    std_input = np.std(input_data, axis=0)
    input_data = (input_data - ave_input) / std_input
    correct_data = np.zeros((n_data, 40))
    for i in range(n_data):
        correct_data[i, correct[i]] = 1

    index = np.arange(n_data)
    index_train = index[index%2 == 0]
    index_test = index[index%2 != 0]
    input_train = input_data[index_train]
    correct_train = correct_data[index_train]
    input_test = input_data[index_test, :]
    correct_test = correct_data[index_test, :]
    n_train = len(index_train)
    n_test = len(index_test)
    index_train = np.arange(n_train)
    index_test = np.arange(n_test)

    error_train_list = []
    error_test_list = []
    accuracy_train_list = []
    accuracy_test_list = []

    for i in range(EPOCH):
        np.random.shuffle(index_train)
        
        error_train = 0
        accuracy_train = 0
        error_test = 0
        accuracy_test = 0
        start = 0
        log_flag = False

        if i % LOG_INTERVAL == 0:
            log_flag = True

        network.set_train_mode(True)
        while start < n_train:
            input = input_train[index_train[start:start+BATCH], :]
            output = correct_train[index_train[start:start+BATCH], :]
            inference = network.forward(input)
            network.backward(output)
            network.update()
            
            if log_flag:
                error_train += -np.sum(output * np.log(inference + 1e-7))
                accuracy_train += np.sum(output.argmax(axis=1) == inference.argmax(axis=1))

            start += BATCH

        network.set_train_mode(False)
        if log_flag:
            start = 0
            while start < n_test:
                input = input_test[index_test[start:start+BATCH], :]
                output = correct_test[index_test[start:start+BATCH], :]
                inference = network.forward(input)
                error_test += -np.sum(output * np.log(inference + 1e-7))
                accuracy_test += np.sum(output.argmax(axis=1) == inference.argmax(axis=1))
                start += BATCH

            error_train /= n_train
            error_test /= n_test
            accuracy_train /= n_train
            accuracy_test /= n_test

            error_train_list.append(error_train)
            error_test_list.append(error_test)
            accuracy_train_list.append(accuracy_train)
            accuracy_test_list.append(accuracy_test)

            print(f'Epoch: {i}/{EPOCH}')
            print(f'Error Train: {error_train}')
            print(f'Error Test: {error_test}')
            print(f'Accuracy Train: {accuracy_train}')
            print(f'Accuracy Test: {accuracy_test}')

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(error_train_list)
    ax1.plot(error_test_list)
    ax2.plot(accuracy_train_list)
    ax2.plot(accuracy_test_list)
    fig.tight_layout()
    plt.show()


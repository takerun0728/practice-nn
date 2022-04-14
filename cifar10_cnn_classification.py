from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from neural_lib import *
import pickle

DEF_MID_NEURON_NUM = 50
EPOCH = 100
LOG_INTERVAL = 1
BATCH = 64
DEF_FLT_NUM = 5
DEF_FLT_SIZE = 3
DEF_ACT_FUNC = 'relu'
DEF_WB_WIDTH = 0.1
DEF_OPT = 'momentum'
DEF_OPT_PARAM = {'eta':0.01, 'rho':0.9, 'alpha':0.9}
DEF_LAM = 0.1
DEF_DROPOUT = 0

class ImageClassificationNetwork(AbstractNetwork):
    def __init__(self, img_ch, img_h, img_w, n_flt=DEF_FLT_NUM, flt_h=DEF_FLT_SIZE, flt_w=DEF_FLT_SIZE,
                 mid_neuron_num=DEF_MID_NEURON_NUM, activate_func=DEF_ACT_FUNC,
                 optimizer=DEF_OPT, opt_params=DEF_OPT_PARAM, lam=DEF_LAM, dropout=DEF_DROPOUT):
        super().__init__()
        self.layers = [ConvLayer(img_ch, img_h, img_w, n_flt, flt_h, flt_w, 1, 1, activate_func=activate_func, optimizer=optimizer, opt_params=opt_params, lam=lam)]
        self.layers.append(PoolingLayer(self.layers[0].y_ch, self.layers[0].y_h, self.layers[0].y_w, 2, 0))
        self.layers.append(Flatten())
        self.layers.append(Layer(self.layers[1].out_len, mid_neuron_num, activate_func=activate_func, optimizer=optimizer, opt_params=opt_params, lam=lam))
        self.layers.append(OutputLayer(mid_neuron_num, 10, activate_func='softmax', opt_params=opt_params, lam=lam))

if __name__ == '__main__':
    network = ImageClassificationNetwork(3, 32, 32)

    input_train = []
    labels_train = []
    for i in range(1,6):
        fp = open(f'cifar-10/data_batch_{i}', 'rb')
        fc = pickle.load(fp, encoding='bytes')
        input_train.extend(fc[b'data'])
        labels_train.extend(fc[b'labels'])
        pass
    fp = open(f'cifar-10/test_batch', 'rb')
    fc = pickle.load(fp, encoding='bytes')
    input_test = fc[b'data']
    labels_test = fc[b'labels']

    n_train = len(labels_train)
    n_test = len(labels_test)

    input_train = np.array(input_train, dtype=np.float32) / 255
    input_test = np.array(input_test, dtype=np.float32) / 255
    input_train = input_train.reshape(n_train, 32, 32, 3)
    input_train = input_train.transpose(0, 3, 1, 2)
    input_test = input_test.reshape(n_test, 32, 32, 3)
    input_test = input_test.transpose(0, 3, 1, 2)

    correct_train = np.zeros((n_train, 10))
    for i in range(n_train):
        correct_train[i, labels_train[i]] = 1

    correct_test = np.zeros((n_test, 10))
    for i in range(n_test):
        correct_test[i, labels_test[i]] = 1
    
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


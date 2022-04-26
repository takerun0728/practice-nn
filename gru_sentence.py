import matplotlib.pyplot as plt
from neural_lib import *

GPU = True
if GPU:
    import cupy as np
else:
    import numpy as np

DEF_MID_NEURON_NUM = 128
DEF_OPT = 'momentum'
DEF_OPT_PARAM = {'eta':0.01, 'rho':0.9, 'alpha':0.9}
DEF_LAMBDA = 0.0
EPOCH = 100000
LOG_INTERVAL = 1
BATCH = 128
N_TIME = 20
NOISE_AMP = 0.1
TEST_LENGTH = 1000
EPS = 1e-7

class GRUNetwork(AbstractNetwork):
    def __init__(self, in_num, out_num,  mid_neuron_num=DEF_MID_NEURON_NUM, optimizer='momentum', opt_params=DEF_OPT_PARAM, lam=DEF_LAMBDA):
        super().__init__()
        self.layers = [GRULayer(in_num, mid_neuron_num, optimizer=optimizer, opt_params=opt_params, lam=lam)]
        self.layers.append(OutputLayer(mid_neuron_num, out_num, activate_func='softmax', opt_params=opt_params, lam=lam))

if __name__ == '__main__':
    with open("kaijin20.txt", mode='r', encoding='utf-8') as f:
        text = f.read()
    print(f"文字数：{len(text)}")

    chars_list = sorted(list(set(text)))
    n_chars = len(chars_list)
    print(f"文字数(重複無){n_chars}")

    network = GRUNetwork(n_chars, n_chars)

    char_to_index = {}
    index_to_char = {}

    for i, char in enumerate(chars_list):
        char_to_index[char] = i
        index_to_char[i] = char

    seq_chars = []
    next_chars = []
    for i in range(0, len(text) - N_TIME):
        seq_chars.append(text[i:i+N_TIME])
        next_chars.append(text[i+N_TIME])

    input_data = np.zeros((len(seq_chars), N_TIME, n_chars), dtype=bool)
    correct_data = np.zeros((len(seq_chars), n_chars), dtype=bool)

    for i, chars in enumerate(seq_chars):
        correct_data[i, char_to_index[next_chars[i]]] = 1
        for j, char in enumerate(chars):
            input_data[i, j, char_to_index[char]] = 1
    pass

    indexes = np.arange(0, len(seq_chars))
    for i in range(EPOCH):
        network.set_train_mode(True)
        error = 0
        start = 0
        log_flag = False
        
        if i % LOG_INTERVAL == 0:
             log_flag = True
        
        while start < len(seq_chars):
            x = input_data[indexes[start:start+BATCH]]
            y_label = correct_data[indexes[start:start+BATCH]]
            start += BATCH
            y_inference = network.forward(x)    
            network.backward(y_label)
            network.update()

            if log_flag:
                error += -np.sum(y_label * np.log(y_inference + EPS))

        network.set_train_mode(False)
        if log_flag:
            print(f'Epoch: {i}/{EPOCH}')
            print(f'Error: {error / len(seq_chars)}')

            predicted = input_data[0]
            for i in range(TEST_LENGTH):
                x = np.array(predicted[-N_TIME:, :]).reshape(1, N_TIME, -1)
                y = network.forward(x)
                predicted = np.vstack((predicted, y))

            test_indexes = np.argmax(predicted, axis=1)
            if GPU:
                test_indexes = test_indexes.get()
            test_sentence = ""
            for idx in test_indexes:
                test_sentence += index_to_char[idx]
            print(test_sentence)

    



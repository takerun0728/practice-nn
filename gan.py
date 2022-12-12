from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from neural_lib import *

MID_NEURON_NUM1 = 32
MID_NEURON_NUM2 = 64
EPOCH = 10010
LOG_INTERVAL = 10
DRAW_INTERVAL = 1000
BATCH = 32
EPS = 1e-7
N_NOISE = 16

class Generator(AbstractNetwork):
    def __init__(self, n_in, n_out, optimizer='momentum', opt_params={'eta':0.001, 'rho':0.9, 'alpha':0.9}, lam=0.0, dropout=0.0):
        super().__init__()
        self.layers = [Layer(n_in, MID_NEURON_NUM1, activate_func='relu', optimizer=optimizer, opt_params=opt_params, lam=lam)]
        self.layers.append(Layer(MID_NEURON_NUM1, MID_NEURON_NUM2, activate_func='relu', optimizer=optimizer, opt_params=opt_params, lam=lam))
        self.layers.append(Layer(MID_NEURON_NUM2, n_out, activate_func='sigmoid', opt_params=opt_params, lam=lam))

class Discriminator(AbstractNetwork):
    def __init__(self, n_in, optimizer='momentum', opt_params={'eta':0.001, 'rho':0.9, 'alpha':0.9}, lam=0.0, dropout=0.0):
        super().__init__()
        self.layers = [Layer(n_in, MID_NEURON_NUM2, activate_func='relu', optimizer=optimizer, opt_params=opt_params, lam=lam)]
        self.layers.append(Layer(MID_NEURON_NUM2, MID_NEURON_NUM1, activate_func='relu', optimizer=optimizer, opt_params=opt_params, lam=lam))
        self.layers.append(OutputLayer(MID_NEURON_NUM1, 1, activate_func='sigmoid', opt_params=opt_params, lam=lam))

def calc_cross_entropy(t, y):
    return -np.sum(t * np.log(y + EPS) + (1 - t) * np.log(1 - y + EPS))

if __name__ == '__main__':
    didits = datasets.load_digits()
    input_data = didits.data / 16
    n_data = len(input_data)
    img_dim = input_data.shape[1]
    idxes = np.arange(n_data)
    generator = Generator(N_NOISE, img_dim)
    discriminator = Discriminator(img_dim)


    error_train_list = []
    for i in range(EPOCH):
        np.random.shuffle(idxes)
        
        error_discriminator = 0
        error_generator = 0
        start = 0
        log_flag = False

        if i % LOG_INTERVAL == 0:
            log_flag = True

        generator.set_train_mode(True)
        while start < n_data:
            #Train Discriminator
            input_one = input_data[idxes[start:start+BATCH], :]
            batch = len(input_one)
            input_zero = generator.forward(np.random.normal(0, 1, (batch, N_NOISE)))
            input_one = input_data[idxes[start:start+batch], :]
            input = np.vstack([input_zero, input_one])
            label = np.hstack([np.zeros(batch), np.ones(batch)]).reshape(-1, 1)
            inference = discriminator.forward(input)
            discriminator.backward(label)
            discriminator.update()

            if log_flag:
                error_discriminator += calc_cross_entropy(label, inference)

            #Train Generator
            img_fake = generator.forward(np.random.normal(0, 1, (batch * 2, N_NOISE)))
            label = np.ones(batch * 2).reshape(-1, 1)
            inference = discriminator.forward(img_fake)
            grad = discriminator.backward(label)
            generator.backward(grad)
            generator.update()

            if log_flag:
                error_generator += calc_cross_entropy(label, inference)
            start += BATCH

        generator.set_train_mode(False)
        generator.set_train_mode(False)
        if log_flag:
            error_discriminator /= n_data * 2
            error_generator /= n_data * 2

            print(f'Epoch: {i}/{EPOCH}')
            print(f'Error discriminator: {error_discriminator}')
            print(f'Error generator: {error_generator}')

        if i % DRAW_INTERVAL == 0:
            fig = plt.figure()
            inference = generator.forward(np.random.normal(0, 1, (64, N_NOISE)))
            for j in range(1, 65):
                
                ax = fig.add_subplot(8, 8, j)
                ax.imshow(inference[j-1].reshape(8, 8))
            plt.show()



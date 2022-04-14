import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(_, y):
    return (1 - y) * y

def relu(x):
    return np.where(x <= 0, 0, x)

def relu_deriv(x, _):
    return np.where(x <= 0, 0, 1)

def leaky_relu(x):
    return np.where(x <= 0, 0.01 * x, x)

def leaky_relu_deriv(x, _):
    return np.where(x <= 0, 0.01, 1)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x, y):
    return 1 - y**2

def identify(x):
    return x

def identify_deriv(_, __):
    return 1

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def softmax_deriv(_, __):
    #This is not implemented currently
    pass

def set_activate_function(activate_func):
    if activate_func == 'sigmoid':
        return sigmoid, sigmoid_deriv
    elif activate_func == 'identify':
        return identify, identify_deriv
    elif activate_func == 'relu':
        return relu, relu_deriv
    elif activate_func == 'leaky_relu':
        return leaky_relu, leaky_relu_deriv
    elif activate_func == 'softmax':
        return softmax, softmax_deriv
    elif activate_func == 'tanh':
        return tanh, tanh_deriv
    
def set_optimizer(optimizer, opt_params):
    if optimizer == 'sgd':
        return SGD(opt_params)
    if optimizer == 'momentum':
        return Momentum(opt_params)
    if optimizer == 'adagrad':
        return AdaGrad(opt_params)
    if optimizer == 'rmsprop':
        return RMSProp(opt_params)

def im2col(images, flt_h, flt_w, out_h, out_w, stride, pad):
    n_bt, n_ch, _, _ = images.shape
    img_pad = np.pad(images, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    cols = np.zeros((n_bt, n_ch, flt_h, flt_w, out_h, out_w))
    for h in range(flt_h):
        for w in range(flt_w):
            cols[:, :, h, w, :, :] = img_pad[:, :, h:h+out_h*stride:stride, w:w+out_w*stride:stride]
        
    cols = cols.transpose(1, 2, 3, 0, 4, 5).reshape(n_ch * flt_h * flt_w, n_bt * out_h * out_w)
    return cols

def col2im(cols, img_shape, flt_h, flt_w, out_h, out_w, stride, pad):
    n_bt, n_ch, img_h, img_w = img_shape
    cols = cols.reshape(n_ch, flt_h, flt_w, n_bt, out_h, out_w).transpose(3, 0, 1, 2, 4, 5)
    images = np.zeros((n_bt, n_ch, img_h + 2 * pad + stride - 1, img_w + 2 * pad + stride - 1))
    
    for h in range(flt_h):
        for w in range(flt_w):
            images[:, :, h:h+out_h*stride:stride, w:w+out_w*stride:stride] += cols[:, :, h, w, :, :]
        
    return images[:, :, pad:img_h+pad, pad:img_w+pad]

class SGD:
    def __init__(self, opt_params):
        self.eta = opt_params['eta']
    
    def __call__(self, grad):
        return -self.eta * grad

class Momentum:
    def __init__(self, opt_params):
        self.eta = opt_params['eta']
        self.alpha = opt_params['alpha']
        self.delta = 0

    def __call__(self, grad):
        self.delta = -(1 - self.alpha) * self.eta * grad + self.alpha * self.delta
        return self.delta

class AdaGrad:
    def __init__(self, opt_params):
        self.eta = opt_params['eta']
        self.h = 0
    
    def __call__(self, grad):
        self.h = self.h + grad**2
        return -self.eta * grad / np.sqrt(self.h)

class RMSProp:
    def __init__(self, opt_params):
        self.eta = opt_params['eta']
        self.rho = opt_params['rho']
        self.h = 0
    
    def __call__(self, grad):
        self.h = self.rho * self.h + (1 - self.rho) * grad**2
        return -self.eta * grad / np.sqrt(self.h)

class Layer:
    def __init__(self, n_upper, n, activate_func='sigmoid', optimizer='sgd', opt_params={'eta':0.1}, lam=0):
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
        self.b = np.zeros(n)
        self.activate_func, self.activate_deriv = set_activate_function(activate_func)
        self.optimizer_w = set_optimizer(optimizer, opt_params)
        self.optimizer_b = set_optimizer(optimizer, opt_params)
        self.opt_params = opt_params
        self.lam = lam

    def forward(self, x, _):
        self.x = x
        self.u = x @ self.w + self.b
        self.y = self.activate_func(self.u)
        return self.y

    def backward(self, grad_y):
        delta = self.set_delta(grad_y, self.u, self.y)
        self.grad_w = self.x.T @ delta + self.w * self.lam
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = delta @ self.w.T
        return self.grad_x

    def set_delta(self, grad_y, u, y):
        return grad_y * self.activate_deriv(u, y)

    def update(self):
        self.w += self.optimizer_w(self.grad_w)
        self.b += self.optimizer_b(self.grad_b)

class OutputLayer(Layer):
    def set_delta(self, t, _, __):
        return self.y - t

class Dropout:
    def __init__(self, dropout_ratio):
        self.ratio = dropout_ratio
    
    def forward(self, x, is_train):
        if is_train:
            rand = np.random.rand(*x.shape)
            self.dropout = np.where(rand > self.ratio, 1, 0)
            return x * self.dropout
        else:
            return x * (1 - self.ratio)
    
    def backward(self, grad_y):
        return grad_y * self.dropout

    def update(self):
        pass

class ConvLayer(Layer):
    def __init__(self, x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad, wb_width=0.01, activate_func='sigmoid', optimizer='sgd', opt_params={'eta':0.1}, lam=0):
        self.y_ch = n_flt
        self.y_h = (x_h - flt_h + 2 * pad) // stride + 1
        self.y_w = (x_w - flt_w + 2 * pad) // stride + 1
        self.out_len = self.y_ch * self.y_h * self.y_w
        self.params = (x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad, self.y_ch, self.y_h, self.y_w)
        self.w = wb_width * np.random.randn(n_flt, x_ch * flt_h * flt_w)
        self.b = wb_width * np.random.randn(1, n_flt)

        self.activate_func, self.activate_deriv = set_activate_function(activate_func)
        self.optimizer_w = set_optimizer(optimizer, opt_params)
        self.optimizer_b = set_optimizer(optimizer, opt_params)
        self.opt_params = opt_params
        self.lam = lam

    def forward(self, x, _):
        n_bt = x.shape[0]
        x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad, y_ch, y_h, y_w = self.params
        
        self.cols = im2col(x, flt_h, flt_w, y_h, y_w, stride, pad)
        u = (self.w @ self.cols).T + self.b
        self.u = u.reshape(n_bt, y_h, y_w, y_ch).transpose(0, 3, 1, 2).reshape(n_bt, y_ch, y_h, y_w)
        self.y = self.activate_func(self.u)
        return self.y

    def backward(self, grad_y):
        n_bt = grad_y.shape[0]
        x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad, y_ch, y_h, y_w = self.params

        delta = self.set_delta(grad_y, self.u, self.y)
        delta = delta.transpose(0, 2, 3, 1).reshape(n_bt * y_h * y_w, y_ch)

        self.grad_w = (self.cols @ delta).T.reshape(n_flt, x_ch * flt_h * flt_w) + self.w * self.lam
        self.grad_b = np.sum(delta, axis=0)

        grad_cols = delta @ self.w
        self.grad_x = col2im(grad_cols.T, (n_bt, x_ch, x_h, x_w), flt_h, flt_w, y_h, y_w, stride, pad)

        return self.grad_x

class PoolingLayer:
    def __init__(self, x_ch, x_h, x_w, pool, pad):
        self.y_ch = x_ch
        self.y_h = x_h // pool if x_h % pool == 0 else x_h // pool +1
        self.y_w = x_w // pool if x_w % pool == 0 else x_w // pool +1
        self.out_len = self.y_ch * self.y_h * self.y_w
        self.params = (x_ch, x_h, x_w, pool, pad, self.y_ch, self.y_h, self.y_w)

    def forward(self, x, _):
        n_bt = x.shape[0]
        x_ch, x_h, x_w, pool, pad, y_ch, y_h, y_w  = self.params
        cols = im2col(x, pool, pool, y_h, y_w, pool, pad)
        cols = cols.T.reshape(n_bt * y_h * y_w * x_ch, pool * pool)
        self.max_index = np.argmax(cols, axis=1).reshape(-1)
        self.y = cols[np.arange(n_bt * y_h * y_w * x_ch), self.max_index]
        self.y = self.y.reshape(n_bt, y_h, y_w, x_ch).transpose(0, 3, 1, 2)
        return self.y

    def backward(self, grad_y):
        n_bt = grad_y.shape[0]
        x_ch, x_h, x_w, pool, pad, y_ch, y_h, y_w  = self.params
        
        grad_cols = np.zeros((pool*pool, grad_y.size))
        grad_cols[self.max_index, np.arange(grad_y.size)] = grad_y.reshape(-1)
        grad_cols = grad_cols.reshape(pool, pool, n_bt, y_ch, y_h, y_w)
        grad_cols = grad_cols.transpose(3, 0, 1, 2, 4, 5)
        grad_cols = grad_cols.reshape(y_ch * pool * pool, n_bt * y_h * y_w)
        self.grad_x = col2im(grad_cols, (n_bt, x_ch, x_h, x_w), pool, pool, y_h, y_w, pool, pad)
        return self.grad_x

    def update(self):
        pass

class Flatten:
    def forward(self, x, _):
        self.shape = x.shape
        return x.reshape(self.shape[0], -1)
    
    def backward(self, grad_y):
        return grad_y.reshape(self.shape)

    def update(self):
        pass

class RNNLayer(Layer):
    def __init__(self, n_upper, n, activate_func='tanh', optimizer='sgd', opt_params={'eta':0.1}, lam=0, is_out_mult=False):
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
        self.v = np.random.randn(n, n) / np.sqrt(n)
        self.b = np.zeros(n)
        self.activate_func, self.activate_deriv = set_activate_function(activate_func)
        self.optimizer_w = set_optimizer(optimizer, opt_params)
        self.optimizer_v = set_optimizer(optimizer, opt_params)
        self.optimizer_b = set_optimizer(optimizer, opt_params)
        self.opt_params = opt_params
        self.lam = lam
        self.is_out_mult = is_out_mult
        self.reset()

    def forward(self, x, is_train): 
        y = np.zeros((x.shape[0], self.b.size))
        x_list = x.transpose(1, 0, 2)
        if is_train or self.is_out_mult:
            self.ys = [y]
        if is_train:    
            self.us = []
            self.xs = x_list
        
        for x_tmp in x_list:
            u = x_tmp @ self.w + y @ self.v + self.b
            y = self.activate_func(u)
            if is_train:
                self.us.append(u)
                self.ys.append(y)
        if self.is_out_mult:
            return np.array(self.ys[1:]).transpose(1, 0, 2)
        else:
            return y

    def calc_grad(self, grad, u, y, x, y_prev):
        delta = self.set_delta(grad, u, y)
        self.grad_w += x.T @ delta + self.w * self.lam
        self.grad_v += y_prev.T @ delta + self.v * self.lam
        self.grad_b += np.sum(delta, axis=0)
        grad_x = delta @ self.w.T
        grad_y = delta @ self.v.T
        return grad_x, grad_y

    def backward(self, grad):
        grad_o = 0
        if self.is_out_mult:
            grad = grad.transpose(1, 0, 2)
            grad_xs = []
            for x, u, y, y_prev, grad_out in zip(self.xs[::-1], self.us[::-1], self.ys[-1::-1], self.ys[-2::-1], grad[::-1]):
                grad_o += grad_out
                grad_x, grad_o = self.calc_grad(grad_o, u, y, x, y_prev)
                grad_xs.append(grad_x)
                y_prev = y
            return np.array(grad_xs)
        else:
            grad_o = grad
            for x, u, y, y_prev in zip(self.xs[::-1], self.us[::-1], self.ys[-1::-1], self.ys[-2::-1]):
                grad_x, grad_o = self.calc_grad(grad_o, u, y, x, y_prev)
                y_prev = y
            return grad_x

    def reset(self):
        self.grad_w = np.zeros_like(self.w)
        self.grad_v = np.zeros_like(self.v)
        self.grad_b = np.zeros_like(self.b)

    def update(self):
        super().update()
        self.v += self.optimizer_v(self.grad_v)
        self.reset()

class LSTMLayer(Layer):
    def __init__(self, n_upper, n, optimizer='sgd', opt_params={'eta':0.1}, lam=0, is_out_mult=False):
        self.w = np.random.randn(4, n_upper, n) / np.sqrt(n_upper)
        self.v = np.random.randn(4, n, n) / np.sqrt(n)
        self.b = np.zeros((4, 1, n))
        self.optimizer_w = set_optimizer(optimizer, opt_params)
        self.optimizer_v = set_optimizer(optimizer, opt_params)
        self.optimizer_b = set_optimizer(optimizer, opt_params)
        self.opt_params = opt_params
        self.lam = lam
        self.is_out_mult = is_out_mult
        self.reset()

    def forward(self, x, is_train): 
        y = np.zeros((x.shape[0], self.b.size))
        x_list = x.transpose(1, 0, 2)
        if is_train or self.is_out_mult:
            self.ys = [y]
        if is_train:    
            self.xs = x_list
            self.gates = []
            self.cs = [y] #the shape of c is same as y
        
        for x_tmp in x_list:
            u = x_tmp @ self.w + y @ self.v + self.b
            a0 = sigmoid(u[0])
            a1 = sigmoid(u[1])
            a2 = tanh(u[2])
            a3 = sigmoid(u[3])
            c = a0 * c + a1 * a2
            y = a3 * np.tanh(c)
            if is_train:
                self.ys.append(y)
                self.gates.append((a0, a1, a2, a3))
                self.cs.append(c)
        if self.is_out_mult:
            return np.array(self.ys[1:]).transpose(1, 0, 2)
        else:
            return y

    def calc_grad(self, grad_o, grad_c, y, x, y_prev, gate, c, c_prev):
        a0, a1, a2, a3 = gate
        tanh_c = np.tanh(c)
        r = grad_c + (grad_o * a3) * tanh_deriv(0, tanh_c)
        delta_a0 = r * c_prev * sigmoid_deriv(0, a0)
        delta_a1 = r * a2 * sigmoid_deriv(0, a1)
        delta_a2 = r * a1 * tanh_deriv(0, a2)
        delta_a3 = grad_o * tanh_c * sigmoid_deriv(0, a3)
        deltas = np.stack((delta_a0, delta_a1, delta_a2, delta_a3))
        self.grad_w += x.T @ deltas
        self.grad_v += y_prev.T @ deltas
        self.grad_b += np.sum(deltas, axis=1)
        grad_x = deltas, self.w.transpose(0, 2, 1).sum(axis=0)
        grad_y = deltas, self.v.transpose(0, 2, 1).sum(axis=0)
        grad_c = r * a0

        return grad_x, grad_y, grad_c

    def backward(self, grad):
        grad_o = 0
        grad_c = 0
        if self.is_out_mult:
            grad = grad.transpose(1, 0, 2)
            grad_xs = []
            for x, y, y_prev, gate, c, c_prev, grad_out in zip(self.xs[::-1], self.ys[-1::-1], self.ys[-2::-1], self.gates, self.cs[-1::-1], self.cs[-2::-1], grad[::-1]):
                grad_o += grad_out
                grad_x, grad_o, grad_c = self.calc_grad(grad_o, grad_c, y, x, y_prev, gate, c, c_prev)
                grad_xs.append(grad_x)
                y_prev = y
            return np.array(grad_xs)
        else:
            grad_o += grad
            for x, y, y_prev, gate, c, c_prev in zip(self.xs[::-1], self.ys[-1::-1], self.ys[-2::-1], self.gates, self.cs[-1::-1], self.cs[-2::-1], grad[::-1]):
                grad_x, grad_o, grad_c = self.calc_grad(grad_o, grad_c, y, x, y_prev, gate, c, c_prev)
                y_prev = y
            return self.grad_x

    def reset(self):
        self.grad_w = np.zeros_like(self.w)
        self.grad_v = np.zeros_like(self.v)
        self.grad_b = np.zeros_like(self.b)

    def update(self):
        super().update()
        self.v += self.optimizer_v(self.grad_v)
        self.reset()

class AbstractNetwork():
    def __init__(self):
        self.is_train = False

    def set_train_mode(self, is_train):
        self.is_train = is_train

    def forward(self, x):
        tmp = x
        for layer in self.layers:
            tmp = layer.forward(tmp, self.is_train)
        return tmp

    def backward(self, t):
        tmp = t
        for layer in self.layers[::-1]:
            tmp = layer.backward(tmp)

    def update(self):
        for layer in self.layers:
            layer.update()

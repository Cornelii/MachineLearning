import numpy as np


def sigmoid(x):
    if type(x) == list:
        x = np.array(x)
    return 1 / (1 + np.exp(-x))


def relu(x):
    if type(x) == list:
        x = np.array(x)
    return np.maximum(0, x)


def softmax(x):
    if type(x) == list:
        x = np.array(x)
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def cross_entropy_error(y, t):
    if type(y) != np.ndarray:
        y = np.array(y)
    if y.ndim == 1:
        y = np.reshape(1, y.shape[0])
    delta = 1e-7
    return -np.sum(t*np.log(y+delta)) / y.shape[0]


def mean_square_error(y, t):
    if type(y) != np.ndarray:
        y = np.array(y)
    return 0.5 * np.sum((y-t)**2)


def numerical_diff(f, x):
    if type(x) != np.ndarray:
        x = np.array(x)
    h = 5e-5
    return (f(x+h) - f(x-h)) / (2*h)


def numerical_gradient(f, x):
    h = 5e-5
    if type(x) != np.ndarray:
        x = np.array(x)
    return (f(x+h) - f(x-h)) / (2*h)


class Relu:

    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, dout):
        pass


class Sigmoid:

    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, dout):
        pass


class Dense:

    def __init__(self, num_input, num_output, activation_layer=None):
        self.x_len = num_input
        self.y_len = num_output
        self.W = np.ones([num_input, num_output], np.float64)
        self.B = np.zeros([1, num_output], np.float64)
        if activation_layer:
            if activation_layer == "sigmoid":
                self.activation_layer = Sigmoid()
            elif activation_layer == "relu":
                self.activation_layer = Relu()
            else:
                raise Exception("Undefined Activation Function!")
        else:
            self.activation_layer = None
        self.dW = np.zeros_like(self.W) # initialization necessary
        self.dB = np.zeros_like(self.B)
        self.x = None

        # initialization
        # normal distribution
    def forward(self, x):
        self.x = x
        y = np.dot(x, self.W) + self.B
        if self.activation_layer:
            y = self.activation_layer.forward(y)
        return y

    def backward(self, dout):
        if self.activation_layer:
            dout = self.activation_layer.backward(dout)

        self.dW = np.dot(self.x.T, dout)
        self.dB = np.sum(dout, axis=1)

        return np.dot(dout, self.W.T)


class Soft_Max_With_Loss:

    def __init__(self):
        pass

    def forward(self, x, t):
        pass

    def backward(self, dout):
        pass


class Normalization_Layer:

    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, dout):
        pass


class Drop_Out:

    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, dout):
        pass


class Optimizer:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def update(self, dw, w):
        w -= self.learning_rate * dw


class Momentum(Optimizer):

    def __init__(self, learning_rate=0.001, alpha=0.99):
        super().__init__(learning_rate)
        self.alpha = 0.99

    def update(self, dw, w):
        pass



















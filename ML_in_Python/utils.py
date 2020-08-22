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


class Dense:

    def __init__(self, num_input, num_output, activation_layer=sigmoid):
        self.x_len = num_input
        self.y_len = num_output
        self.W = np.ones([num_input, num_output], np.float64)
        self.B = np.zeros([1, num_output], np.float64)
        self.activation_layer = activation_layer
        self.dW = np.zeros_like(self.W) # initialization necessary
        self.dB = np.zeros_like(self.B)

        # initialization
        # normal distribution
    def forward(self, x):
        y = np.dot(x, self.W) + self.B
        return self.activation_layer(y)

    def backward(self, y):
        pass

    def get_gradients(self, y):
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



















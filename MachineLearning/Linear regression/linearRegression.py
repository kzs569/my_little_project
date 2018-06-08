from sklearn.linear_model import LinearRegression
from MachineLearning.Others import ActivationFunctions
import numpy as np


class LinearRegression:

    def __init__(self, weight=0.0, bias=0.0, learning_rate=0.001, num_iter=1000):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.weight = weight
        self.bias = bias

    def fit(self):
        pass

    def predict(self):
        pass

    def compute_gradient(self, b_current, m_current, data, learning_rate):
        b_gradient = 0
        m_gradient = 0

        N = float(len(data))
        # Two ways to implement this
        # first way
        # for i in range(0,len(data)):
        #     x = data[i,0]
        #     y = data[i,1]
        #
        #     #computing partial derivations of our error function
        #     #b_gradient = -(2/N)*sum((y-(m*x+b))^2)
        #     #m_gradient = -(2/N)*sum(x*(y-(m*x+b))^2)
        #     b_gradient += -(2/N)*(y-((m_current*x)+b_current))
        #     m_gradient += -(2/N) * x * (y-((m_current*x)+b_current))

        # Vectorization implementation
        x = data[:, 0]
        y = data[:, 1]
        b_gradient = -(2 / N) * (y - m_current * x - b_current)
        b_gradient = np.sum(b_gradient, axis=0)
        m_gradient = -(2 / N) * x * (y - m_current * x - b_current)
        m_gradient = np.sum(m_gradient, axis=0)
        # update our b and m values using out partial derivations

        new_b = b_current - (learning_rate * b_gradient)
        new_m = m_current - (learning_rate * m_gradient)
        return [new_b, new_m]

    def compute_error(self, b, m, data):

        totalError = 0
        # Two ways to implement this
        # first way
        # for i in range(0,len(data)):
        #     x = data[i,0]
        #     y = data[i,1]
        #
        #     totalError += (y-(m*x+b))**2

        # second way
        x = data[:, 0]
        y = data[:, 1]
        totalError = (y - m * x - b) ** 2
        totalError = np.sum(totalError, axis=0)

        return totalError / float(len(data))

    def optimizer(self, data, starting_b, starting_m, learning_rate, num_iter):
        b = starting_b
        m = starting_m

        # gradient descent
        for i in range(num_iter):
            # update b and m with the new more accurate b and m by performing
            # thie gradient step
            b, m = self.compute_gradient(b, m, data, learning_rate)
            if i % 100 == 0:
                print('iter {0}:error={1}'.format(i, self.compute_error(b, m, data)))
        return [b, m]


if __name__ == 'main':
    lr = LinearRegression()

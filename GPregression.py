import numpy as np
from scipy import linalg

class GPRegression(object):

    def __init__(self, kernel, beta):
        self.kernel = kernel
        self.beta = beta


    def fit(self, x, y):

        self.X = x
        self.Y = y

        self.K = self.kernel(x, x)
        C = self.K + (1/self.beta)*np.eye(len(x))

        self.C_inv = linalg.inv(C)

    def predict(self, x):
        k = self.kernel(x, self.X)
        c = self.Kernel(x, x) + (1/self.beta)

        mu = k*self.C_inv*self.Y
        var = c - k*self.C_inv*k

        return mu, var

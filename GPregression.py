import numpy as np
from scipy import linalg

class GPregression(object):

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
        c = np.diag(self.kernel(x, x)) + (1/self.beta)
        mu = k.T.dot(self.C_inv.dot(self.Y))
        var = c - k.T.dot(self.C_inv.dot(k))
        var = np.diag(var)
        return mu, var

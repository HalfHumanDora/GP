import numpy as np

class GPRegression(object):

    def __init__(self, kernel):
        self.kernel = kernel
        self.beta = beta


    def fit(self, x, y):

        self.X = x
        self.Y = y

        C = self.kernel(x, x)


    def predict(self, x):
        k = self.kernel(x, self.X)
        c = self.Kernel(x, x) + (1/self.beta)

        mu = k*self.C_inv*self.Y
        var = c - k*self.C_inv*k

        return mu, var

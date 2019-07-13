import numpy as np
from sklearn.gaussian_process.kernels import RBF


class GP_Regressor(object):
    def __init__(self, kernel=None):
        self.kernel = kernel

    def fit(self, X, y):
        K = self.kernel(X)
        print(K)


        pass


    def predict(self, X):
        pass



if __name__ == "__main__":
    X = np.array([0.2, 0.4, 0.6])
    y = [1, 0.9, 0.8]


    kernel = RBF()
    gp = GP_Regressor(kernel=kernel)
    gp.fit(X, y)

import numpy as np
from sklearn.gaussian_process.kernels import RBF

# from kernels import Linear, RBF

class GP_Regressor(object):
    def __init__(self, kernel):
        self.kernel = kernel

    def fit(self, X, y):
        K = self.kernel(X, X)
        # Get inverse matrix of K : K^-1
        inv_K = np.linalg.inv(K)

        self.inv_K = inv_K
        self.train_X = X
        self.train_y = y

    def predict(self, X):
        # X is batch.
        N = len(self.train_X)
        M = len(X)
        k_star = np.zeros(shape=(N, M))

        k_star = self.kernel(self.train_X, X)
        k_star_star = self.kernel(X)

        pred_mean = np.dot(np.dot(k_star.T, self.inv_K), self.train_y)
        pred_std = k_star_star - np.dot(np.dot(k_star.T, self.inv_K), k_star)

        return pred_mean, pred_std



if __name__ == "__main__":
    X = np.array([[0.2], [0.4], [0.6], [0.8], [1.0]])
    y = np.array([[1], [0.8], [0.6], [0.4], [0.2]])

    kernel = RBF()
    gp = GP_Regressor(kernel=kernel)
    gp.fit(X, y)

    given = np.array([[0.7]])
    mean, std = gp.predict(given)
    print("prediction mean:\n{}\n Cov:\n{}".format(mean, std))

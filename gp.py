import numpy as np
from sklearn.gaussian_process.kernels import RBF


class GP_Regressor(object):
    def __init__(self, kernel=None):
        self.kernel = kernel

    def fit(self, X, y):
        K = self.kernel(X, X)
        # Get inverse matrix of K : K^-1
        inv_K = np.linalg.inv(K)

        self.inv_K = inv_K
        self.train_X = X
        self.train_y = y

    def predict(self, x):
        # x is one data

        k_star = [self.kernel(x, [train_x])[0] for train_x in self.train_X]
        k_star = np.asarray(k_star)
        k_star_star = self.kernel(x)

        pred_mean = np.dot(np.dot(k_star.T, self.inv_K), y)
        pred_std = k_star_star - np.dot(np.dot(k_star.T, self.inv_K), k_star)

        return pred_mean, pred_std



    def predict_batch(self, X):
        # X is batch data

        N = len(self.train_X)
        M = len(X)
        k_star = self.kernel(self.train_X, X)
        k_star_star = self.kernel(X)




if __name__ == "__main__":
    X = np.array([[0.2], [0.4], [0.6]])
    y = [1, 0.9, 0.8]

    kernel = RBF()
    gp = GP_Regressor(kernel=kernel)
    gp.fit(X, y)

    mean, std = gp.predict_batch([[0.5], [0.1]])
    print("prediction mean:{} std:{}".format(mean, std))

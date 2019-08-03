import numpy as np
from sklearn.gaussian_process.kernels import RBF
from collections import defaultdict
# from kernels import Linear, RBF

class GpRegressor(object):
    def __init__(self, kernel):
        self.kernel = kernel

        self.train_X = None

    def fit(self, X, y):
        K = self.kernel(X)
        # Get inverse matrix of K : K^-1
        inv_K = np.linalg.inv(K)

        self.inv_K = inv_K
        self.train_X = X
        self.train_y = y

    def predict(self, X):
        if self.train_X is not None:

            # X is batch.
            N = len(self.train_X)
            M = len(X)
            k_star = np.zeros(shape=(N, M))

            k_star = self.kernel(self.train_X, X)
            k_star_star = self.kernel(X)

            pred_mean = np.dot(np.dot(k_star.T, self.inv_K), self.train_y)
            pred_std = k_star_star - np.dot(np.dot(k_star.T, self.inv_K), k_star)

        elif self.train_X is None:
            pred_mean = np.zeros(shape=np.array(X).shape)
            # print(pred_mean, X)
            k_star_star = self.kernel(X)
            pred_std = k_star_star

        return pred_mean, pred_std

class GpClassifier(object):
    def __init__(self, kernel):
        self.kernel = kernel

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

class BayesianOptimization(object):
    def __init__(self, kernel, model, search_params):
        self.kernel = kernel

        self.model = model

        # search_params is dictionary
        # key : name of param
        # value : search range
        for k, v in search_params.items():
            search_params[k] = np.array(v)
        self.search_params = search_params

        self.best_candidates = defaultdict(tuple)
        self.searched_idx = defaultdict(set)

        self.iter = 0

        self.train_X = []
        self.train_y = []

    def get_ucb(self, X):
        # calculate Upper Confidence bound of each input
        pred_mean, pred_std = self.gp_regressor.predict(X)

        return pred_mean + pred_std * np.sqrt(np.log(self.iter)/self.iter)

    def search(self, search_iter=100):
        gp_regressor = GpRegressor(kernel=self.kernel)
        self.gp_regressor = gp_regressor


        for i in range(1, search_iter+1, 1):
            self.iter = i
            print("{}th iteration..".format(i))

            for param, param_range in self.search_params.items():
                

                ucb_scores = self.get_ucb(param_range)[0]
                print(ucb_scores)

                target_idx = np.argmax(ucb_scores)
                x = param_range[target_idx]
                y = self.model(x)

                self.train_X.append(x)
                self.train_y.append(y)

                if len(self.best_candidates[param]) == 0:
                    self.best_candidates[param] = (x, y)
                elif self.best_candidates[param][1] < y:
                    self.best_candidates[param] = (x, y)

                self.gp_regressor.fit(self.train_X, self.train_y)







if __name__ == "__main__":
    # X = np.array([[0.2], [0.4], [0.6], [0.8], [1.0]])
    # y = np.array([[1], [0.8], [0.6], [0.4], [0.2]])
    #
    kernel = RBF()
    # gp = GpRegressor(kernel)
    # gp.fit(X, y)
    #
    # given = np.array([[0.7], [0.44]])
    # mean, std = gp.predict(given)
    # print("prediction mean:\n{}\n Cov:\n{}".format(mean, std))

    def sin_curve(x):
        y_org = np.sin(x)
        np.random.seed(0)
        y0 = np.sin(x) + (np.random.rand(1)/5 - 0.5/5)
        return y0

    x_range = [[i]for i in np.linspace(-1,1,50)]
    search_params = {"x": x_range}

    bo = BayesianOptimization(kernel, sin_curve, search_params)
    bo.search(100)

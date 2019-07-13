import numpy as np

class Kernel(object):
    def __init__(self,*param):
        self.param = list(param)

    def __call__(self, x1, x2):
        raise NotImplementedError()

class Linear(Kernel):
    def __call__(self, x1, x2):
        # print(x1, x2)
        # x1[:,None]*np.tile(x2, (,1))
        return x1.dot(x2.T)

class RBF(Kernel):
    def __call__(self, x1, x2):
        # import IPython
        # IPython.embed()
        sub = x2.dot(x1.T)/x2-x2
        sub = sub**2
        return np.exp(-0.5*sub)

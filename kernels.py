import numpy as np

class Kernel(object):
    def __init__(self,*param):
        self.param = list(param)

    def __call__(self, x1, x2):
        raise NotImplementedError()

class LinearKernel(Kernel):

    def __call__(self, x1, x2):
        return x1*x2

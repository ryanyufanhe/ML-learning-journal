from model.weight import weight_initializer


class LinearModel(object):
    def __init__(self, dim, loss, grad_fn, is_reg=False):
        self.dim = dim
        self.loss = loss
        self.grad_fn = grad_fn
        self.is_reg = is_reg
        self.weights = weight_initializer(self.dim, mode='normal')

    def fit(self, X, Y):
        pass
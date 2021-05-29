import numpy as np
from functools import lru_cache

class svm():

    def __init__(self,
                 kernel="rbf", lmd=1e-1, gamma=0.1, bias=1.0, max_iter=100):
        if kernel == "rbf":
            self.kernel = self.gaussian_kernel(gamma=gamma)
        else:
            self.kernel = self.linear_kernel
        self.kernel = kernel_func
        self.lmd = lmd
        self.bias = bias
        self.max_iter = max_iter

    def linear_kernel(self,x, y):
        return np.dot(x, y)

    def gaussian_kernel(self, x, y, gamma):
        diff = x - y
        return np.exp(-gamma * np.dot(diff, diff))

    def fit(self, X, y):
        X_with_bias = np.c_[X, np.ones((np.shape(X)[0])) * self.bias]
        self.y = y
        data_size, feature_size = np.shape(X)
        alpha = np.zeros((data_size, 1))
        for t in range(1, self.max_iter + 1):
            new_alpha = np.copy(alpha)
            it = np.random.randint(low=0, high=data_size)
            x_it = X_with_bias[it]
            y_it = y[it]
            res = 0
            for x_j, alpha_j in zip(X_with_bias, alpha):
                res += alpha_j * y_it * self.kernel(x_it, x_j)
            if (y_it * (1. / (self.lmd * t))) * res < 1:
                new_alpha[it] += 1
            alpha = new_alpha
        self.alpha = alpha
        return alpha

    def score(self, X):
        X_with_bias = np.c_[X, np.ones((np.shape(X)[0])) * self.bias]
        y_score = []
        for x in X_with_bias:
            i = 0
            for (x_j, y_j, alpha_j) in zip(self.X_with_bias, self.y, self.alpha):
                i += alpha_j * y_j * self.kernel(x_j, x)
            y_score.append((1. / (self.lmd * self.max_iter)) * i)
        return score

    def predict(self,X):
        y_score = self.score(X)
        y_predict = []
        for s in y_score:
            if s >= 0.:
                y_predict.append(1)
            else:
                y_predict.append(-1)
        return y_predict

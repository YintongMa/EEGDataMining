import numpy as np

X = np.array([
    [-2,4,-1],
    [4,1,-1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],

])

y = np.array([-1,-1,1,1,1])
class svm():

    def __init__(self,
                 kernel="linear", lmd=1e-1, gamma=0.1, bias=1.0, max_iter=100):
        if kernel not in self.__kernel_dict:
            print(kernel + " kernel does not exist!\nUse rbf kernel.")
            kernel = "rbf"
        if kernel == "rbf":
            def kernel_func(x, y):
                return self.__kernel_dict[kernel](x, y, gamma=gamma)
        else:
            kernel_func = self.__kernel_dict[kernel]
        self.kernel = kernel_func
        self.lmd = lmd
        self.bias = bias
        self.max_iter = max_iter

    def __linear_kernel(x, y):
        return np.dot(x, y)

    def __gaussian_kernel(x, y, gamma):
        diff = x - y
        return np.exp(-gamma * np.dot(diff, diff))

    __kernel_dict = {"linear": __linear_kernel, "rbf": __gaussian_kernel}

    def train(self,X, Y):
        w = np.zeros(len(X[0]))
        eta = 1
        epochs = 100000

        for epoch in range(1, epochs):
            for i, x in enumerate(X):
                if (Y[i] * np.dot(X[i], w)) < 1:
                    w = w + eta * ((X[i] * Y[i]) + (-2 * (1 / epoch) * w))
                else:
                    w = w + eta * (-2 * (1 / epoch) * w)

        return w

        return w
        # X_count = X.shape[0]
        # alpha = np.zeros(X_count)
        # eta = 0.1
        # flag = 1
        # max_iterations = 1000
        # for ite in range(max_iterations):
        #
        #     for i in range(X.shape[0]):
        #         sum = 0
        #         for j in range(X.shape[0]):
        #             val = alpha[j] * Y[j] * self.linear_kernel(X[i], X[j])
        #             sum = sum + val
        #         if sum <= 0:
        #             sum = -1
        #         elif sum > 0:
        #             sum = 1
        #         if Y[i] != sum:
        #             alpha[i] = alpha[i] + 1
        # return alpha

    def fit(self, X, y):
        def update_alpha(alpha, t):
            data_size, feature_size = np.shape(self.X_with_bias)
            new_alpha = np.copy(alpha)
            it = np.random.randint(low=0, high=data_size)
            x_it = self.X_with_bias[it]
            y_it = self.y[it]
            if (y_it * (1. / (self.lmd * t)) * sum([alpha_j * y_it * self.kernel(x_it, x_j) for x_j, alpha_j in zip(self.X_with_bias, alpha)])) < 1.:
                new_alpha[it] += 1
            return new_alpha

        self.X_with_bias = np.c_[X, np.ones((np.shape(X)[0])) * self.bias]
        self.y = y
        alpha = np.zeros((np.shape(self.X_with_bias)[0], 1))

        for t in range(1, self.max_iter + 1):
            alpha = update_alpha(alpha, t)
        self.alpha = alpha
        return alpha

    def decision_function(self, X):
        X_with_bias = np.c_[X, np.ones((np.shape(X)[0])) * self.bias]
        y_score = [(1. / (self.lmd * self.max_iter)) *
                   sum([alpha_j * y_j * self.kernel(x_j, x)
                        for (x_j, y_j, alpha_j) in zip(
                        self.X_with_bias, self.y, self.alpha)])
                   for x in X_with_bias]
        return np.array(y_score)

    def predict(self, X):
        y_score = self.decision_function(X)
        y_predict = map(lambda s: 1 if s >= 0. else -1, y_score)
        return y_predict

if __name__ == "__main__":

    cf = svm()

    print(cf.train(X,y))
    print(cf.fit(X, y))
    print (cf.predict(X))
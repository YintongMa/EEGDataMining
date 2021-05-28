import numpy as np
from functools import lru_cache

class svm():

    def __init__(self,
                 kernel="rbf", lmd=1e-1, gamma=0.1, bias=1.0, max_iter=100):
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

    def fit(self, X, y):
        def update_alpha(alpha, t):
            data_size, feature_size = np.shape(self.X_with_bias)
            new_alpha = np.copy(alpha)
            it = np.random.randint(low=0, high=data_size)
            x_it = self.X_with_bias[it]
            y_it = self.y[it]

            # alpha[k] = alpha[k] + eta[k] * (1 - myData.loc[k, 2] * sum(alpha * myData.loc[:, 2] * K[:, k]))
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


    def predict(self,X):
        X_with_bias = np.c_[X, np.ones((np.shape(X)[0])) * self.bias]

        y_score = []

        for x in X_with_bias:
            i = 0
            for (x_j, y_j, alpha_j) in zip(self.X_with_bias, self.y, self.alpha):
                i += alpha_j * y_j * self.kernel(x_j, x)
            y_score.append((1. / (self.lmd * self.max_iter)) * i)

        y_predict = []
        for s in y_score:
            if s >= 0.:
                y_predict.append(1)
            else:
                y_predict.append(-1)
        return y_predict

if __name__ == "__main__":



    data = np.load("eeg_data.npz")
    X = data['x']
    y = data['y']
    print(y.shape)
    ds = []
    for i in X:
        ds.append(np.nan_to_num(i.flatten()))

    print(np.array(ds).shape)

    train_x = np.array(ds)
    train_y = [1 for i in range(len(y))]
    for i in range(len(y)):
        if y[i] == -1:
            train_y[i] = -1
    train_y = np.array(train_y)
    print(train_y.shape)

    from sklearn.svm import SVC

    cf = svm()
    cf.fit(train_x, train_y)
    p = cf.predict((train_x[:200]))

    cnt = 0.
    for i,j in zip(p,train_y[:200]):
        print(i,j)
        if i == j:
            cnt += 1

    print("cnt",cnt)

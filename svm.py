import numpy as np
from functools import lru_cache

# class KP:
#     def __init__(self):
#         self._x = None
#         self._alpha = self._b = self._kernel = None
#
#     @staticmethod
#     def _poly(x, y, p=4):
#         return (x.dot(y.T) + 1) ** p
#
#     @staticmethod
#     def _rbf(x, y, gamma):
#         return np.exp(-gamma * np.sum((x[..., None, :] - y) ** 2, axis=2))
#
#     def fit(self, x, y, kernel="rbf", p=None, gamma=None, c=1, lr=0.0001, batch_size=128, epoch=10000):
#         x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
#         if kernel == "poly":
#             p = 4 if p is None else p
#             self._kernel = lambda x_, y_: self._poly(x_, y_, p)
#         elif kernel == "rbf":
#             gamma = 1 / x.shape[1] if gamma is None else gamma
#             self._kernel = lambda x_, y_: self._rbf(x_, y_, gamma)
#         else:
#             raise NotImplementedError("Kernel '{}' has not defined".format(kernel))
#         self._alpha = np.zeros(len(x))
#         self._b = 0.
#         self._x = x
#         k_mat = self._kernel(x, x)
#         k_mat_diag = np.diag(k_mat)
#         for _ in range(epoch):
#             self._alpha -= lr * (np.sum(self._alpha * k_mat, axis=1) + self._alpha * k_mat_diag) * 0.5
#             indices = np.random.permutation(len(y))[:batch_size]
#             k_mat_batch, y_batch = k_mat[indices], y[indices]
#             err = 1 - y_batch * (k_mat_batch.dot(self._alpha) + self._b)
#             if np.max(err) <= 0:
#                 continue
#             mask = err > 0
#             delta = c * lr * y_batch[mask]
#             self._alpha += np.sum(delta[..., None] * k_mat_batch[mask], axis=0)
#             self._b += np.sum(delta)
#
#     def predict(self, x, raw=False):
#         x = np.atleast_2d(x).astype(np.float32)
#         k_mat = self._kernel(self._x, x)
#         y_pred = self._alpha.dot(k_mat) + self._b
#         if raw:
#             return y_pred
#         return np.sign(y_pred).astype(np.float32)

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


    # xc, yc = gen_two_clusters()
    # xc, yc = train_x, train_y
    # kp = KP()
    # kp.fit(xc, yc, kernel = "rbf", p=1)
    # print("准确率：{:8.6} %".format((kp.predict(xc[:100,]) == yc[:100]).mean() * 100))
    #


    # print(cf.train(X,y))
    # print(cf.fit(train_x, train_y))

    # from sklearn.svm import SVC
    # cf = SVC()
    # cf.fit(train_x, train_y)
    # p = cf.predict(train_x[12000:,])
    # cnt = 0.
    # for i,j in zip(p,train_y[12000:]):
    #     print(i,j)
    #     if i == j:
    #         cnt += 1
    #
    # print("cnt",cnt)
    # print(p)
    # print("The score of linear is : %f" % cf.score(train_x,train_y))

    #
    # cf = CustomSVM()
    # cf.fit(train_x[12000:], train_y[12000:])
    # p = cf.predict((train_x[12000:]))
    #
    # cnt = 0.
    # for i,j in zip(p,train_y[12000:]):
    #     print(i,j)
    #     if i == j:
    #         cnt += 1
    #
    # print("cnt",cnt)

import numpy as np
from functools import lru_cache

class svm_new():
    def __init__(self, kernel="rbf", lmd=1e-1, sigma=0.1, max_iter=100):
        self.kernel = kernel
        self.lmd = lmd
        self.sigma = sigma
        self.max_iter = max_iter
        self.alpha = 0
        self.w = 0


    def kernel_matrix(self, X):
        if self.kernel == "linear":
            return X.dot(X.T)
        elif self.kernel == "rbf":
            return X.dot(X.T)
        else:
            row, col = X.shape
            GassMatrix = np.zeros(shape=(row, row))
            i = 0
            for v_i in X:
                j = 0
                for v_j in X:
                    GassMatrix[i, j] = Gaussian(v_i.T, v_j.T, self.sigma)
                    j += 1
                i += 1
            return GassMatrix

    def Gaussian(self, x, z):
        norm = np.linalg.norm(x - z)**2
        up = ((norm**2)*-1)
        down = (2 * (self.sigma ** 2))
        return math.exp(up/down)

    def KRR(self, K,y):
        m,n = K.shape
        return np.linalg.inv(K + np.dot(self.lmd,np.identity(m))).dot(y)


    def fit(self, X, y, lamb=0.01, alpha_init=None, eta=0.01, max_step=1000):
        """
        Compute the optimal alpha for an SVM using the kernel matrix K
        and observations y.
        alpha_init : our initial value for alpha
        eta: step size
        """

        K = self.kernel_matrix(X)
        n = len(y)
        if alpha_init is None:
            alpha_init = np.zeros(n)
        assert K.shape == (n, n)
        assert len(alpha_init) == n
        # make sure all labels are +/ - 1 and that we have
        # both positive and negative labels
        assert (np.unique(y) == np.array([-1, 1])).all()
        alpha_k = alpha_init
        for k in range(max_step):
            print(k)
            alpha_old = alpha_k
            ## your code here to update alpha_k
            res = np.zeros(n)
            for i in range(n):
                if y[i] * K[:, i].T.dot(alpha_old) < 1:
                    res += np.identity(n).dot(-y[i] * K[:, i])
            res += 2 * lamb * K.dot(alpha_old)

            alpha_k = alpha_old - eta * res
            # # compute y_est
            # y_est = (K + lamb * np.identity(n)).dot(alpha_k)
            # accuracy = np.mean((y_est > 0) == (y > 0))
            alpha_norm_change = np.linalg.norm(alpha_k - alpha_old)
            # print("k = {:5}".format(k) + ",alpha_norm_change = {:3.2f}".format(alpha_norm_change) +
            #       ",accuracy = {:3.1f}".format(accuracy * 100))
            if alpha_norm_change < 1e-1:
                break
        self.alpha = alpha_k
        self.w = X.T.dot(alpha_k)
        return alpha_k, w

    def predict (self, X):
        y_predict = []
        for s in X.dot(self.w.T):
            if s[0] >= 0.:
                y_predict.append(1)
            else:
                y_predict.append(-1)
        return np.array(y_predict)


class svm():

    def __init__(self,
                 kernel="rbf", lmd=1e-1, gamma=0.1, bias=1.0, max_iter=100):
        if kernel == "rbf":
            print("use rbf kernel")
            self.kernel = self.__gaussian_kernel
        else:
            print("use linear kernel")
            self.kernel = self.__linear_kernel

        self.lmd = lmd
        self.bias = bias
        self.gamma = gamma
        self.max_iter = max_iter

    def __linear_kernel(self, x, y):
        return np.dot(x, y)

    def __gaussian_kernel(self,x, y):
        diff = x - y
        return np.exp(-self.gamma * np.dot(diff, diff))


    def fit(self, X, y):
        def update_alpha(alpha, t):
            data_size, feature_size = np.shape(self.X_with_bias)
            new_alpha = np.copy(alpha)
            it = np.random.randint(low=0, high=data_size)
            x_it = self.X_with_bias[it]
            y_it = self.y[it]
            res = 0
            k = 0
            for x_j, alpha_j in zip(self.X_with_bias, alpha):
                print(k)
                k += 1
                res += alpha_j * y_it * self.kernel(x_it, x_j)
            if (y_it * (1. / (self.lmd * t)) * res) < 1.:
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
import numpy as np
import pandas as pd
import sklearn

from sklearn.decomposition import PCA
from sklearn.utils import shuffle

# Load data with specific id number
def load_data(id_num):
    data = np.load("/content/eeg_data.npz")
    X = data['x']
    y = data['y']

    index = [i for i in range(len(y)) if y[i] == id_num]

    output_data = []
    output_label = []

    for i in index:
        output_data.append(X[i])
        output_label.append(y[i])

    return output_data, output_label


# Compare seeing one number with rest
def binary_all_channel(data, label, id_num):
    if len(data) != len(label):
        print("Something is wrong here")
        return

    output_data = []
    output_label = []

    for i in range(len(label)):
        if label[i] != id_num and label[i] != -1:
            print("Something is wrong here")
            break
        if label[i] != -1:
            output_label.append([1])
        else:
            output_label.append([-1])

        feature = np.concatenate(data[i])
        feature = np.nan_to_num(feature)
        output_data.append(feature)

    return output_data, output_label


def predict(X, w, mode):
    raw_val = np.dot(X.transpose(), w)

    if mode == "binary":
        if raw_val >= 0:
            return 1
        if raw_val < 0:
            return -1
    if mode == "multiclass":
        return round(raw_val[0])


def cross_val(X, y, batch_size):
    error_arr = []
    subset_num = int(len(X) / batch_size) - 1
    for i in range(subset_num):
        print("batch: " + str(i))
        error = 0
        X_test = X[i * batch_size: (i + 1) * batch_size]
        y_test = y[i * batch_size: (i + 1) * batch_size]
        X_train = np.concatenate((X[0: i * batch_size], X[(i + 1) * batch_size: len(X)]))
        y_train = np.concatenate((y[0: i * batch_size], y[(i + 1) * batch_size: len(y)]))

        print('running svm linear')
        cf = svm(kernel="linear")
        cf.fit(X_train, y_train)

        print('predicting')
        result = cf.predict((X_test))
        for j in range(len(result)):
            if result[j] != y_test[j]:
                error = error + 1
        error_rate = error / batch_size
        error_arr.append(error_rate)

        print("error rate is " + str(error_rate))
        print()

        error = 0
        print('running svm rbf')
        cf = svm(kernel="rbf")
        cf.fit(X_train, y_train)

        print('predicting')
        result = cf.predict((X_test))
        for j in range(len(result)):
            if result[j] != y_test[j]:
                error = error + 1
        error_rate = error / batch_size

        print("error rate is " + str(error_rate))
        print()

        error = 0
        print('running base svm linear')
        from sklearn.svm import SVC
        cf_base = SVC(kernel="linear")
        cf_base.fit(X_train, y_train)
        print('predicting')
        result = cf_base.predict((X_test))
        for j in range(len(result)):
            if result[j] != y_test[j]:
                error = error + 1
        error_rate = error / batch_size
        print("base error rate is " + str(error_rate))
        print()

        error = 0
        print('running base svm rbf')
        from sklearn.svm import SVC
        cf_base = SVC(kernel="rbf")
        cf_base.fit(X_train, y_train)
        print('predicting')
        result = cf_base.predict((X_test))
        for j in range(len(result)):
            if result[j] != y_test[j]:
                error = error + 1
        error_rate = error / batch_size
        print("base error rate is " + str(error_rate))
        print()

    print ("Error rate of each iteration: " + str(error_arr))
    print ("Average error rate:" + str(np.average(error_arr)))
def compute_pca(data):
    pca = PCA()
    pca_data = pca.fit_transform(data)
    return pca_data
if __name__ == "__main__":

    # data = np.load("eeg_data.npz")
    # data_x = data['x']
    # data_y = data['y']
    #
    # X = []
    # y = []
    #
    # for i in range(len(data_y)):
    #     if data_y[i] != -1:
    #         y.append(1)
    #     else:
    #         y.append(-1)
    #
    # for i in data_x:
    #     X.append(np.nan_to_num(i.flatten()))
    #
    # X_normalized = sklearn.preprocessing.normalize(X, norm='l2')
    #
    # X_pca = compute_pca(X_normalized)
    # X_pca = X_normalized

    # all_data, all_label = shuffle(X_pca, y)
    # cross_val(all_data, all_label, 1240)
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

    cf = svm_new(kernel="linear")
    # cf = svm(kernel="linear")
    cf.fit(train_x, train_y)
    #
    # from sklearn.svm import SVC
    #
    # cf = svm(kernel="linear")
    # cf.fit(train_x, train_y)
    # p = cf.predict((train_x[:200]))
    #
    # cnt = 0.
    # for i,j in zip(p,train_y[:200]):
    #     print(i,j)
    #     if i == j:
    #         cnt += 1
    #
    # print("cnt",cnt)




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

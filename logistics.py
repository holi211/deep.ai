import numpy as np
import matplotlib.pylab as plt
import h5py

class Logistics():
    # 数据预处理
    def LoadDataset(self):
        train_dataset = h5py.File('train_catvnoncat.h5', "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

        test_dataset = h5py.File('test_catvnoncat.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])

        classes = np.array(test_dataset["list_classes"][:])

        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

    def DatePreocess(self):
        train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = self.LoadDataset()
        train_set_x = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1)).T
        test_set_x = test_set_x_orig.reshape((test_set_x_orig.shape[0], -1)).T
        # 归一化
        train_set_x = train_set_x / 255
        test_set_x = test_set_x / 255
        return train_set_x, train_set_y, test_set_x, test_set_y, classes

    def Sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def CostFunction(self, x, y, w, b):
        count = x.shape[1]
        A = self.Sigmoid(np.dot(w.T, x)+b)
        cost = (-1 / count) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
        dw = (1 / count) * np.dot(x, (A - y).T)
        db = (1 / count) * np.sum(A - y)
        return cost, dw, db

    def Gradient(self, x, y, w, b, alpha, maxIteration, print_cost=False):
        costs = []
        for i in range(maxIteration):
            cost, dw, db = self.CostFunction(x, y, w, b)
            w = w - alpha * dw
            b = b - alpha * db
            if (print_cost) and (i % 100 == 0):
                costs.append(cost)
                print("迭代的次数: %i ， 误差值： %f" % (i, cost))
        return w, b, costs

    def Predict(self, w, b, x):
        count = x.shape[1]
        Y_predict = np.zeros((count,1))
        A = self.Sigmoid(np.dot(w.T, x) + b)
        A = A.T
        for i in range(count):
            Y_predict[i] = 1 if A[i] > 0.5 else 0
        return Y_predict

if __name__ == '__main__':
    lis = Logistics()
    train_set_x, train_set_y, test_set_x, test_set_y, classes = lis.DatePreocess()
    w = np.random.randn(12288,1)
    b = 0
    w, b ,costs = lis.Gradient(train_set_x, train_set_y, w, b, 0.001, 10000, print_cost = True)
    Y_predict = lis.Predict(w, b, test_set_x)
    Y_predicty = lis.Predict(w, b, train_set_x)
    print('准确率=' + format(100 - np.mean(np.abs(Y_predict - test_set_y.T)) * 100))
    print('准确率=' + format(100 - np.mean(np.abs(Y_predicty - train_set_y.T)) * 100))





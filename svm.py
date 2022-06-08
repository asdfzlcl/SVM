from numpy import *
import time
import matplotlib.pyplot as plt


class Svm:
    def __init__(self, kernel_type, C, kernel_parameter):
        self.kernel = kernel_type
        self.c = C
        self.parameter = kernel_parameter
        self.data=[]
        self.label=[]
        self.alpha=[]
        self.b = 0
        self.N = 0
        print("SVM创建完成")

    def kernel_mul(self, x1, x2):
        # 线性核
        if self.kernel == "linear":
            return x1.dot(x2.T)

    def loaddate(self, data_test, label):
        self.data = data_test.copy()
        self.label = label.copy()
        self.N = self.data.shape[0]
        self.alpha = zeros(self.N)
        print("数据导入完成")

    def limit_range(self, A, H, L):
        if A > H:
            return H
        if A < L:
            return L
        return A


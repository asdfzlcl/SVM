import numpy as np
import time
import matplotlib.pyplot as plt
import math

class Svm:
    def __init__(self, kernel_mul_type, C, kernel_mul_parameter):
        self.kernel = kernel_mul_type
        self.C = C
        self.parameter = kernel_mul_parameter
        self.data = []
        self.label = []
        self.alpha = []
        self.b = 0
        self.N = 0
        self.E = []
        print("SVM创建完成")

    def kernel_mul(self, x1, x2):
        # 线性核
        if self.kernel == "linear":
            return x1.dot(x2.T)
        if self.kernel == "poly":
            return (x1.dot(x2.T)*self.parameter[0]+self.parameter[1])**self.parameter[2]
        if self.kernel == "gauss":
            return math.exp(-self.parameter[0]*(np.sum((x1-x2)*(x1-x2))))
        if self.kernel == "sigmoid":
            return math.tanh(self.parameter[0]*x1.dot(x2.T)+self.parameter[1])
        print("当前核未知")
        return 0


    def loaddate(self, data_test, label):
        self.data = data_test.copy()
        self.label = label.copy()
        self.N = self.data.shape[0]
        self.alpha = np.zeros(self.N)
        self.E = np.zeros(self.N)
        print("数据导入完成")
        print("共"+str(self.N)+"个数据")

    def limit_range(self, A, H, L):
        if A > H:
            return H
        if A < L:
            return L
        return A

    # 计算期望答案
    def g(self, i):
        gi = self.b
        for j in range(self.N):
            gi += self.alpha[j] * self.label[j] * self.kernel_mul(self.data[i], self.data[j])
        return gi

    # 计算输入数据
    def forecast(self,test):
        gi = self.b
        for j in range(self.N):
            gi += self.alpha[j] * self.label[j] * self.kernel_mul(test, self.data[j])
        if gi>0:
            return 1
        else:
            return 0

    # 检验KKT条件
    def KKT(self, i):
        y_g = self.g(i) * self.label[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return 1 == y_g
        else:
            return y_g <= 1

    # 寻找第一个违反KKT条件的下标
    def find_alpha1(self):
        index_list = [i for i in range(self.N) if 0 < self.alpha[i] < 1]
        non_satisfy_list = [i for i in range(self.N) if i not in index_list]
        index_list.extend(non_satisfy_list)
        # error = np.zeros(self.N)
        candidate = []
        for i in index_list:
            if self.KKT(i):
                continue
            candidate.append(i)
        return candidate

    def updateE(self):
        for i in range(self.N):
            self.E[i] = self.g(i) - self.label[i]

    # 寻找第二个下标
    def find_alpha2(self, alpha1_id):
        if self.E[alpha1_id] > 0:
            return np.argmin(self.E)
        else:
            return np.argmax(self.E)

    def SMO(self, max_turn):
        turn = 0
        while max_turn > 0:
            max_turn -= 1
            turn += 1
            alpha_list = self.find_alpha1()
            print(alpha_list)
            change = 0
            for id1 in alpha_list:
                if self.KKT(id1):
                    continue
                for id2 in range(self.N):
                    self.updateE()
                    if id2 == id1:
                        continue
                    if self.KKT(id1):
                        break
                    # print(id1,id2)
                    E1 = self.E[id1]
                    E2 = self.E[id2]
                    if self.label[id1] == self.label[id2]:
                        L = max(0, self.alpha[id1] + self.alpha[id2] - self.C)
                        H = min(self.C, self.alpha[id1] + self.alpha[id2])
                    else:
                        L = max(0, self.alpha[id2] - self.alpha[id1])
                        H = min(self.C, self.C + self.alpha[id2] - self.alpha[id1])

                    if L==H:
                        self.E[id2] = 1.5*self.E[id1]

                    alpha1_old = self.alpha[id1]
                    alpha2_old = self.alpha[id2]
                    eta = self.kernel_mul(self.data[id1], self.data[id1]) + self.kernel_mul(self.data[id2],
                                                                                    self.data[id2]) - 2 * self.kernel_mul(
                        self.data[id1], self.data[id2])
                    if eta <= 0:
                        continue

                    alpha2_new = self.limit_range(alpha2_old + self.label[id2] * (E1 - E2) / eta, H, L)
                    alpha1_new = self.label[id2] * (alpha2_old - alpha2_new) * self.label[id1] + alpha1_old


                    b1 = -E1 - \
                         self.label[id1] * self.kernel_mul(self.data[id1], self.data[id1]) * (
                            alpha1_new - alpha1_old) - \
                         self.label[id2] * self.kernel_mul(
                        self.data[id2], self.data[id1]) * (alpha2_new - alpha2_old) + self.b

                    b2 = -E2 - \
                         self.label[id1] * self.kernel_mul(self.data[id1], self.data[id2]) * (
                            alpha1_new - alpha1_old) - \
                         self.label[id2] * self.kernel_mul(
                        self.data[id2], self.data[id2]) * (alpha2_new - alpha2_old) + self.b


                    if 0 < self.alpha[id1] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[id2] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1+b2)/2

                    self.alpha[id2] = alpha2_new
                    self.alpha[id1] = alpha1_new
                    change = alpha1_new - alpha1_old
                    # if abs(alpha1_new-alpha1_old)<0.01:
                    #     self.E[id2] = 1.5*self.E[id1]


            if turn%20==0:
                print("第" + str(turn) + "轮完成")
                print(self.alpha)
            if change<0.00001:
                print("结束")
                return 0

        print(self.find_alpha1())



    def support_vector(self):
        support = []
        for i in range(self.N):
            if self.alpha[i]>0:
                support.append(self.data[i])
        return np.array(support)

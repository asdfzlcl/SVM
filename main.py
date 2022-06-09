import svm
import numpy as np
import matplotlib.pyplot as plt
import random

train = []
label = []
x_ = []
x0 = []
y0 = []
x1 = []
y1 = []
maxL = 500
for i in range(50):
    x = random.uniform(0, maxL)
    y = random.uniform(0, maxL)
    if 4*x - 3*y - 200 > 20:
        train.append([x, y])
        label.append(1)
        x1.append(x)
        y1.append(y)
    elif 4*x - 3*y -200 < -20:
        train.append([x, y])
        label.append(-1)
        x0.append(x)
        y0.append(y)



svm1 = svm.Svm("linear",10,[0,1])

data = np.array(train)
label = np.array(label)

svm1.loaddate(data,label)
svm1.SMO(100)
print(svm1.alpha,svm1.b)
# for i in range(9):
#     print(svm1.g(i))


step = 5
x = np.arange(0,maxL+step,step)
y = np.arange(0,maxL+step,step)
X,Y = np.meshgrid(x,y)
print(X.shape)
Z = np.zeros(X.shape)

a = 0
for i in np.arange(0,maxL+step,step):
    b = 0
    for j in np.arange(0,maxL+step,step):
        Z[a][b] = svm1.forecast(np.array([i,j]))
        b += 1
    a += 1
plt.contourf(x,y,Z)
plt.contour(x,y,Z)
plt.scatter(y0, x0, c = "red", marker='o', label='+1')
plt.scatter(y1, x1, c = "green", marker='o', label='-1')

plt.show()
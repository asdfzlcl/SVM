import svm
import numpy as np

svm1 = svm.Svm("linear",2,[0,1])
data = np.array([
    [1,1],
    [1,2],
    [1,3],
    [2,1],
    [2,2],
    [2,3],
    [3,1],
    [3,2],
    [3,3]
])
label = np.array([-1,-1,-1,
                  1,-1,-1,
                  1,1,-1])

svm1.loaddate(data,label)
svm1.SMO(3000)
print(svm1.alpha,svm1.b)
for i in range(9):
    print(svm1.g(i))
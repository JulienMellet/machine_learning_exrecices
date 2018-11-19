import matplotlib.pyplot as plt
import numpy as np
#import scipy

from numpy import matrix, matmul
from numpy.linalg import inv

tmp_m = [[0, 1],
         [1, 3],
         [2, 5],
         [3, 7]]
tmp_p = 1


def calc_a_plus(mat):
    """
   Used to calcul the A+ = (A_T*A)^(-1) where x*=(A+)A_T.b
   :type mat: matrix
   """
    mat_trans = mat.transpose()
    res_matmul = matmul(mat_trans, mat)
    a_plus = inv(res_matmul)
    return a_plus


def line_regression(main_mat, p):
    """
   Used to calcul the main elements of Ax*=b
   :type main_mat: list
   :type p: int
   """
    mat = list()
    vec = list()
    for elem in main_mat:
        tmp = list()
        for j in range(p+1):
            tmp.append(pow(elem[0], j))
        mat.append(tmp)
        vec.append(elem[1])
    #print("a = {}".format(mat))
    #print("b = {}".format(vec))
    a_plus = calc_a_plus(matrix(mat))
    tmp_res = matmul(a_plus, matrix(mat).transpose())
    res = matmul(tmp_res, vec)
    return res

# Draw a sinus
x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, np.sin(x), 'g', label='sin')

# Create a vector y_sinoise of lenght 10 drawing a sinus with noise
y_sinoise = []
x_sinoise = []
for i in range(10):
    y_sinoise.append(np.sin(i/9*2*np.pi) + (np.random.random()-0.5))
    x_sinoise.append(i/9*2*np.pi)
#print(y_sinoise)
#print(x_sinoise)

# Linear regression
A = []
for i in range(10):
    A.append([x_sinoise[i], y_sinoise[i]])
print(A)


# Draw points of y_sinoise
x = np.linspace(0, 2*np.pi, 10)
plt.plot(x, y_sinoise, 'o')

plt.xlabel('x')
plt.ylabel('y')

plt.title("Regression")

plt.legend()

plt.show()

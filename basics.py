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

def poly_fun(y_x):
    res = 0
    for i in range(len(y_x)+1):
        res += y_x.item(i)*pow(x,i)
    return res


###############################################################################
# main
###############################################################################

# Create a vector y_sinoise of lenght 10 drawing a sinus with noise
y_sinoise = list()
x_sinoise = list()
for i in range(10):
    y_sinoise.append(np.sin(i/9*2*np.pi) + (np.random.random()-0.5))
    x_sinoise.append(i/9*2*np.pi)
#print(y_sinoise)
#print(x_sinoise)

# Linear regression
m = list()
for i in range(10):
    m.append([x_sinoise[i], y_sinoise[i]])

#y_1 = poly_fun(line_regression(m,1))
y_1 = line_regression(m,1)
y_2 = line_regression(m,2)
y_3 = line_regression(m,3)
y_4 = line_regression(m,4)
y_9 = line_regression(m,9)


# Draw M = 1
x = np.linspace(0, 2*np.pi, 100)
plt.subplot(211)
plt.plot(x, y_1.item(0) + y_1.item(1)*x, 'r', label='M = 1')

# Draw M = 2
plt.subplot(211)
plt.plot(x,  y_2.item(0) + y_2.item(1)*x + y_2.item(2)*x**2, label='M = 2')

# Draw M = 3
plt.subplot(211)
plt.plot(x,  y_3.item(0) + y_3.item(1)*x + y_3.item(2)*x**2 +  y_3.item(3)*x**3, label='M = 3')
"""
# Draw M = 4
plt.subplot(211)
plt.plot(x,  y_4.item(0) + y_4.item(1)*x + y_4.item(2)*x*x +  y_4.item(3)*x**3 +  y_4.item(4)*x**4, label='M = 4')

# Draw M = 9
plt.subplot(211)
plt.plot(x,  y_9.item(0) + y_9.item(1)*x + y_9.item(2)*x**2 +  y_9.item(3)*x**3 +  y_9.item(4)*x**4 +  y_9.item(5)*x**5 + y_9.item(6)*x**6 + y_9.item(7)*x**7 + y_9.item(8)*x**8 + y_9.item(9)*x**9, label='M = 9')
"""
# Draw a sinus
x = np.linspace(0, 2*np.pi, 100)
plt.subplot(211)
plt.plot(x, np.sin(x), '--g', label='sin')

# Draw points of y_sinoise
x = np.linspace(0, 2*np.pi, 10)
plt.subplot(211)
plt.plot(x, y_sinoise, 'o')

# Error graph
x = np.linspace(0, 2*np.pi, 100)
plt.subplot(212)
plt.plot(x, abs(np.sin(x) - (y_3.item(0) + y_3.item(1)*x + y_3.item(2)*x**2 +  y_3.item(3)*x**3)), label='Error M = 3')

plt.xlabel('x')
plt.ylabel('y')

plt.title("Linear regression")

plt.legend()

plt.show()

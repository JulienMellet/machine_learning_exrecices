import matplotlib.pyplot as plt
import numpy as np
#import scipy

from numpy import matrix, matmul
from numpy.linalg import inv

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
    mat = []
    vec = []
    for elem in main_mat:
        tmp = []
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

def poly_fun(y_x,x):
    res = 0
    for i in range(len(y_x)):
        res += y_x.item(i)*pow(x,i)
        #print(res)
    return res

def f_1(y_1,x):
    return y_1.item(0) + y_1.item(1)*x

def f_2(y_2,x):
    return y_2.item(0) + y_2.item(1)*x + y_2.item(2)*x**2

def f_3(y_3,x):
    return y_3.item(0) + y_3.item(1)*x + y_3.item(2)*x**2 +  y_3.item(3)*x**3

def f_4(y_4,x):
    return y_4.item(0) + y_4.item(1)*x + y_4.item(2)*x**2 +  y_4.item(3)*x**3 +  y_4.item(4)*x**4

def f_5(y_5,x):
    return y_5.item(0) + y_5.item(1)*x + y_5.item(2)*x**2 +  y_5.item(3)*x**3 +  y_5.item(4)*x**4 +  y_5.item(5)*x**5

def f_6(y_6,x):
    return y_6.item(0) + y_6.item(1)*x + y_6.item(2)*x**2 +  y_6.item(3)*x**3 +  y_6.item(4)*x**4 +  y_6.item(5)*x**5 + y_6.item(6)*x**6

def f_7(y_7,x):
    return y_7.item(0) + y_7.item(1)*x + y_7.item(2)*x**2 +  y_7.item(3)*x**3 +  y_7.item(4)*x**4 +  y_7.item(5)*x**5 + y_7.item(6)*x**6 + y_7.item(7)*x**7

def f_8(y_8,x):
    return y_8.item(0) + y_8.item(1)*x + y_8.item(2)*x**2 +  y_8.item(3)*x**3 +  y_8.item(4)*x**4 +  y_8.item(5)*x**5 + y_8.item(6)*x**6 + y_8.item(7)*x**7 + y_8.item(8)*x**8

def f_9(y_9,x):
    return y_9.item(0) + y_9.item(1)*x + y_9.item(2)*x**2 +  y_9.item(3)*x**3 +  y_9.item(4)*x**4 +  y_9.item(5)*x**5 + y_9.item(6)*x**6 + y_9.item(7)*x**7 + y_9.item(8)*x**8 + y_9.item(9)*x**9

# Error between prediction and trained data
def E(y_k, xt, yt):
    res = 0
    for i in range(xt):
        res += (f_k(y_k, xt[i]) - yt[i])**2
    return res/2

###############################################################################
# main
###############################################################################

# Create a vector y_sinoise of lenght 10 drawing a sinus with noise
# It simulate trained data
y_sinoise = []
x_sinoise = []
for i in range(10):
    y_sinoise.append(np.sin(i/9*2*np.pi) + (np.random.random()-0.5))
    x_sinoise.append(i/9*2*np.pi)
#print(y_sinoise)
#print(x_sinoise)
x_error = []
for i in range(9):
    x_error.append(i)
print(x_error)

###############################################################################
# Linear regression
m = []
for i in range(10):
    m.append([x_sinoise[i], y_sinoise[i]])

#y_1 = poly_fun(line_regression(m,1))
y_1 = line_regression(m,1)
y_2 = line_regression(m,2)
y_3 = line_regression(m,3)
y_4 = line_regression(m,4)
y_9 = line_regression(m,9)

plt.figure(1)
# Draw M = 1
x = np.linspace(0, 2*np.pi, 100)
#plt.plot(x, y_1.item(0) + y_1.item(1)*x, 'r', label='M = 1')
plt.plot(x, f_1(y_1,x), 'r', label='M = 1')

# Draw M = 2
plt.plot(x, f_2(y_2,x), label='M = 2')
#plt.plot(x,  poly_fun(y_2[0],x), label='M = 2')

# Draw M = 3
plt.plot(x, f_3(y_3,x), label='M = 3')

# Draw M = 4
plt.plot(x, f_4(y_4,x), label='M = 4')

# Draw M = 9
plt.plot(x, f_9(y_9,x), label='M = 9')

# Draw a sinus
x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, np.sin(x), '--g', label='sin')

# Draw points of y_sinoise
x = np.linspace(0, 2*np.pi, 10)
plt.plot(x, y_sinoise, 'o')

plt.xlabel('x')
plt.ylabel('y')

plt.title("Linear regression")

###############################################################################
# Error graph
plt.figure(2)

y_error = []
for i in range 9

x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, abs(np.sin(x) - (f_3(y_3,x))), label='Error M = 3')

plt.xlabel('x')
plt.ylabel('y')

plt.title("error function")

###############################################################################
# Regularization
plt.figure(3)
# Draw M = 9
plt.plot(x, f_9(y_9,x), label='M = 9')

# Draw a sinus
x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, np.sin(x), '--g', label='sin')

plt.xlabel('x')
plt.ylabel('y')

plt.title("Regularization")

plt.legend()

#plt.show()

import matplotlib.pyplot as plt
import numpy as np
#import scipy

from numpy import matrix, matmul, array
from numpy.linalg import inv

def calc_a_plus(mat):
    """
    Used to calcul the A+ = (A_T*A)^(-1) where x*=(A+)A_T.b
    :type mat: matrix
    """
    mat_trans = mat.transpose()
    res_matmul = matmul(mat_trans, mat) #Add here regularization term
    a_plus = inv(res_matmul)
    return a_plus

def line_regression(mat_M, p):
    """
    Used to calcul the main elements of Ax*=b
    :type main_mat: list
    :type p: int
    """
    mat_A = []
    vec_b = []

    #for elem in range(len(mat_M)):
    for elem in mat_M:
        tmp = []
        for j in range(p+1):
            tmp.append(pow(float(elem[0]), j))
        mat_A.append(tmp)
        vec_b.append(float(elem[1]))
    #print("a = {}".format(mat))
    #print("b = {}".format(vec))
    A_plus = calc_a_plus(array(mat_A))
    tmp_res = matmul(A_plus, matrix(mat_A).transpose())
    res = matmul(tmp_res, vec_b)#.tolist()
    #print('res : {}'.format(res))
    return res

def poly_fun(y_x,x):
    """
    Used to make weighted array into a function
    :type y_x: array
    :type x: array
    """
    res = 0
    X = []

    #print('res:{}'.format(res))
    #print("len y_x", y_x.shape[1])
    for i in range(y_x.shape[1]):
        X.append(np.power(x,i))

    #print("Y = ", y_x)
    #print("X = ", matrix(X))
    res = matmul(y_x, matrix(X))
    #transform res into a list
    Y = list()
    for i in range(len(x)):
        Y.append(res.item(i))
    #print("Y= ", Y)
    return Y

# Error between prediction and trained data
def E(y_k, xt, yt):
    res = 0
    for i in range(len(xt)):
        res += (poly_fun(y_k, xt)[i]-yt[i])**2
    return res/2

def calc_a_plus_regulator(mat, l):
    """
    Used to calcul the A+ = (A_T*A + l*I)^(-1) where x*=(A+)A_T.b
    :type mat: matrix
    """
    #print("len(mat) = ", int(np.sqrt(mat.size)))
    length = int(np.sqrt(mat.size))
    reg = np.exp(l)*np.eye(length)
    #print("reg = ", reg)
    mat_trans = mat.transpose()
    res_matmul = matmul(mat_trans, mat) + reg
    a_plus_reg = inv(res_matmul)
    return a_plus_reg

def regularization(mat_M, l):
    """
    Make calculation of linear regression adding a regularizator term l
    :type main_mat: list
    :type l: float
    """
    mat_A = []
    vec_b = []
    p=9
    #for elem in range(len(mat_M)):
    for elem in mat_M:
        tmp = []
        for j in range(p+1):
            tmp.append(pow(float(elem[0]), j))
        mat_A.append(tmp)
        vec_b.append(float(elem[1]))
    #print("a = {}".format(mat))
    #print("b = {}".format(vec))
    A_plus_regulator = calc_a_plus_regulator(array(mat_A), l)
    tmp_res = matmul(A_plus_regulator, matrix(mat_A).transpose()) + A_plus_regulator
    res = matmul(tmp_res, vec_b)#.tolist()
    #print('res : {}'.format(res))
    return res
from basics import *

# Create a vector y_sinoise of lenght 10 drawing a sinus with noise
# It simulate trained data
y_sinoise = []
x_sinoise = []
for i in range(10):
    y_sinoise.append(np.sin(i/9*2*np.pi) + (np.random.random()-0.5))
    x_sinoise.append(i/9*2*np.pi)
#print(y_sinoise)
#print(x_sinoise)

###############################################################################
# Linear regression
m = []
for i in range(10):
    m.append([x_sinoise[i], y_sinoise[i]])

plt.figure(1)
# Draw M = 1
x = np.linspace(0, 2*np.pi, 100)
#plt.plot(x, y_1.item(0) + y_1.item(1)*x, 'r', label='M = 1')
plt.plot(x, poly_fun(line_regression(m,1),x), 'r', label='M = 1')

# Draw M = 2
plt.plot(x, poly_fun(line_regression(m,2),x), label='M = 2')
#plt.plot(x,  poly_fun(y_2[0],x), label='M = 2')

# Draw M = 3
plt.plot(x, poly_fun(line_regression(m,3),x), label='M = 3')

# Draw M = 4
plt.plot(x, poly_fun(line_regression(m,4),x), label='M = 4')
"""
# Draw M = 5
plt.plot(x, poly_fun(line_regression(m,5),x), label='M = 5')

# Draw M = 6
plt.plot(x, poly_fun(line_regression(m,6),x), label='M = 6')

# Draw M = 7
plt.plot(x, poly_fun(line_regression(m,7),x), label='M = 7')

# Draw M = 8
plt.plot(x, poly_fun(line_regression(m,8),x), label='M = 8')
"""
# Draw M = 9
plt.plot(x, poly_fun(line_regression(m,9),x), label='M = 9')

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
#print("E = ", E(line_regression(m,1), x_sinoise, y_sinoise))

plt.figure(2)

x_error = []
for i in range(8):
    x_error.append(i+1)
#np.array(x_error)
#print("x_error = ", x_error)
"""
y_error = [E(line_regression(m,1), x_sinoise, y_sinoise), E(line_regression(m,2), x_sinoise, y_sinoise), E(line_regression(m,3), x_sinoise, y_sinoise), E(line_regression(m,4), x_sinoise, y_sinoise), E(line_regression(m,5), x_sinoise, y_sinoise), E(line_regression(m,6), x_sinoise, y_sinoise), E(line_regression(m,7), x_sinoise, y_sinoise), E(line_regression(m,8), x_sinoise, y_sinoise), E(line_regression(m,9), x_sinoise, y_sinoise)]
"""
y_error = []
for i in range(8):
    y_error.append(E(line_regression(m,i+1), x_sinoise, y_sinoise))

# Make Erms
for i in range(8):
    y_error[i] = np.sqrt(2*y_error[i]/(i+1))

plt.plot(x_error, y_error, label='Training')

plt.xlabel('M')
plt.ylabel('Erms')

plt.title("Error function")


###############################################################################
# Regularization
plt.figure(3)
# Draw M = 9
x = np.linspace(0, 2*np.pi, 100)
plt.plot(x,  poly_fun(line_regression(m,9),x), label='M = 9')

# Draw Regularized
y_9_r = regularization(m, -100)
#("reg vect = ", y_9_r)

plt.plot(x,  poly_fun(y_9_r,x), label='Regularized')


# Draw a sinus
x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, np.sin(x), '--g', label='sin')

plt.xlabel('x')
plt.ylabel('y')

plt.title("Regularization")

plt.legend()

plt.show()
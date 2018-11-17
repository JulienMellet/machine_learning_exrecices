import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, np.sin(x), 'g', label='sin')


randx = []
for i in range(10):
    randx.append(np.sin(i/9*2*np.pi) + (np.random.random()-0.5))
print(randx)

"""
matx = 1/np.random.randint(low = -10, high = 10, size=(1, 10))
print(matx)
"""

x = np.linspace(0, 2*np.pi, 10)
plt.plot(x, randx , 'o')

plt.xlabel('x label')
plt.ylabel('y label')

plt.title("Simple Plot")

plt.legend()

plt.show()

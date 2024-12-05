import numpy as np
import matplotlib.pyplot as plt
import random
import math 
from scipy.special import logit, expit

random.seed(42)

#Question 3
#Part 3A
n = 10**3
m = 20
b = -3
def r(x_val):
    return m * x_val + b


x_vals = np.array([z/n for z in range(1, n+1)])
y_vals = np.array([r(x) + random.normalvariate(0, 4) for x in x_vals])


plt.scatter(x_vals, y_vals)
x = np.linspace(0, 1, 3)
plt.plot(x, r(x), c = 'r')
plt.show()

#Part 3B
x1_10 = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
y1_10 = [0.70, 8.20, 1.30, 5.30, 3.60, 3.50, 1.80, 6.70, 23.90, 20.80]

def function(x_val, y_val):
    return ((2*math.pi)**0.5 * 4)**-1 * math.exp(-((y_val - r(x_val))**2/(2*16)))

def y1n_g_x1n(x_values, y_values):
    product = 1
    for index, x in enumerate(x_values):
        product = product * function(x, y_values[index])
    return product

print(math.log(y1n_g_x1n(x1_10, y1_10)) * -1)

#Part 4

alpha = -5
beta = [35, -35]
x_vals_p4 = [z/100 for z in range(1, 101)]

def phi(x_val):
    return (x_val, x_val**2)

def logistic1(a, bx):
    return math.exp(a + bx) / (1 +  math.exp(a + bx))

bx_vals= [phi(x4)[0] * beta[0] + phi(x4)[1] * beta[1] for x4 in x_vals_p4]
weights = [logistic1(alpha, bx_val) for bx_val in bx_vals]
y_vals_p4 = np.random.binomial(1, p = weights)


plt.scatter(x_vals_p4, y_vals_p4)
plt.plot(x_vals_p4, weights, c = 'r')
plt.show()
pass

import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.mixture import GaussianMixture as gm
from scipy.stats import norm

with open('/Users/rudeciabernard/Downloads/Dataset1.txt', 'r') as file:
    d = file.read()

with open('/Users/rudeciabernard/Downloads/Dataset2.txt', 'r') as file2:
    d2 = file2.read()

#Skip-- this is just data cleaning
d = d.split('\n')
d.pop()
d2 = d2.split('\n')
d2.pop()

def cleanup(s: str ):
    proto = s.split('  ')
    z = list(filter(lambda s: s != '', proto))
    z[0] = z[0].lstrip()
    return list(map(float, z))

dataset1 = list(map(cleanup, d))
dataset2 = list(map(cleanup, d2))

#Problem 3

all_x_vals = [x[1] for x in dataset1]

z1_x_vals = [x[1] for x in list(filter(lambda s: int(s[0]) == 1, dataset1))]
mean1 = sum(z1_x_vals) / len(z1_x_vals)
var1 = np.var(z1_x_vals)
pi1 = len(z1_x_vals)/len(dataset1)

z2_x_vals = [x[1] for x in list(filter(lambda s: int(s[0]) == 2, dataset1))]
mean2 = sum(z2_x_vals) / len(z2_x_vals)
var2 = np.var(z2_x_vals)
pi2 = len(z2_x_vals)/len(dataset1)

def gaussian(x_val, mean, var):
    return ((2*math.pi)**0.5 * var**0.5)**-1 * np.exp(-((x_val - mean)**2/(2*var)))

 
plotting_vals = [n/10 for n in range(-60, 20)]

mle_comp1 = [gaussian(x, mean1, var1) for x in plotting_vals]
mle_comp2 = [gaussian(x, mean2, var2) for x in plotting_vals]
mle_tot = [pi1 * gaussian(x, mean1, var1) + pi2 * gaussian(x, mean2, var2) for x in plotting_vals]

print(mean1, mean2)
print(var1, var2)
print(pi1, pi2)


plt.hist(all_x_vals, bins = 50, density = True)
plt.show()
plt.plot(plotting_vals, mle_tot)
plt.show()
plt.hist(z1_x_vals, bins = 50, density = True)
plt.show()
plt.plot(plotting_vals, mle_comp1)
plt.show()
plt.hist(z2_x_vals, bins = 50, density = True)
plt.show()
plt.plot(plotting_vals, mle_comp2)
plt.show()


#Problem 4
p4_mod = gm(n_components= 2).fit(dataset2)


x = np.linspace(-4, 4)
y = np.linspace(-4, 4)

X, Y = np.meshgrid(x, y)


z1 = np.log(gaussian(X, p4_mod.means_[0], 1) * p4_mod.weights_[0])
z2 = np.log(gaussian(Y, p4_mod.means_[1], 1) * p4_mod.weights_[1]) 

Z = z1 + z2


fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, levels = 100)
plt.show()

#Problem 5

p5_mod = gm(n_components= 2, init_params= 'random_from_data', random_state= 42).fit(dataset2)
hist2vals = [l[0] for l in dataset2]


x = np.linspace(-4, 4)
y = np.linspace(-4, 4)

z = [gaussian(x_val, p5_mod.means_[0], p5_mod.covariances_[0][0][0]) * p5_mod.weights_[0] + 
    gaussian(x_val, p5_mod.means_[1], p5_mod.covariances_[1][0][0]) * p5_mod.weights_[1] for x_val in x]


plt.hist(hist2vals, bins = 50, density = True)
plt.plot(x, z, c = 'r')
plt.show()

print(p5_mod.means_)
print(p5_mod.weights_)
print(p5_mod.score(dataset2))

pass
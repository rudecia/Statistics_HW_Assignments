from sklearn.mixture import GaussianMixture as gmm
from sklearn.linear_model import LinearRegression, LassoCV, LogisticRegression
import math
import numpy as np
import matplotlib.pyplot as plt

def cleanup(s: str ):
    proto = s.split('  ')
    z = list(filter(lambda s: s != '', proto))
    z[0] = z[0].lstrip()
    return list(map(float, z))

#Dataset 1
with open('/Users/rudeciabernard/Downloads/Dataset1.txt', 'r') as file:
    dataset1 = file.read()

dataset1 = dataset1.split('\n')
dataset1.pop()
dataset1 = list(map(cleanup, dataset1))
dataset1 = np.array([[n[0] - 1, n[1]] for n in dataset1])

#Dataset 3
with open('/Users/rudeciabernard/Downloads/Dataset3.txt', 'r') as file:
    dataset3 = file.read()
    
dataset3 = dataset3.split('\n')
dataset3.pop()
dataset3 = np.array(list(map(cleanup, dataset3)))



#Problem 1: Lasso Regression
d3_x = dataset3[:, 0].reshape(200, 1)
d3_y = dataset3[:, 1].reshape(200, 1)


def phi(x):
    l = [x, x**2]
    for num in range(1, 20):
        l.append(np.sin(x * num))
        l.append(np.cos(x * num))
    return l


p1_xvals = [phi(x_val[0]) for x_val in d3_x]
p1_plot1 = [phi(x) for x in np.linspace(-3, 3, 50)]
p1_plot2 = [phi(x) for x in np.linspace(-4, 4, 50)]


p1_model = LassoCV(cv = 10)
p1_model.fit(p1_xvals, d3_y)

print(f'The alpha for Problem 1 is {p1_model.intercept_}')
print(f'The beta for this model is {p1_model.coef_}')


#On the interval [-3, 3]
plt.plot(np.linspace(-3, 3, 50), p1_model.predict(p1_plot1), c = 'r')
plt.scatter(d3_x, d3_y)
plt.title('Dataset 3: Model on the Interval [-3, 3] (with Lasso)')
plt.show()

#On the interval [-4, 4]
plt.plot(np.linspace(-4, 4, 50), p1_model.predict(p1_plot2), c = 'r')
plt.scatter(d3_x, d3_y)
plt.title('Dataset 3: Model on the Interval [-4, 4] (with Lasso)')
plt.show()


#There are 17 non-zero intercepts according to lasso 



#Problem 2

d1_yvals = dataset1[:, 0].reshape(1000, 1)
d1_xvals = dataset1[:, 1].reshape(1000, 1)

def gaussian(x_val, mean, var):
    return ((2*math.pi)**0.5 * var**0.5)**-1 * np.exp(-((x_val - mean)**2/(2*var)))

def r(x: float):
    return 0.825 * gaussian(x, -0.2920, 0.2097) / (0.825 * gaussian(x, -0.2920, 0.2097) + 0.175 * gaussian(x, -2.4148, 1.8701))

p2_plotx = np.linspace(-6, 2, 100)
p2_ploty = [r(x) for x in p2_plotx]
plt.plot(p2_plotx, p2_ploty, c = 'r')
plt.scatter(d1_xvals, d1_yvals)
plt.title('Dataset 1: E(X|Y) Plot from Gaussian Mixture Model')
plt.show()

#Problem 3
def quad_phi(x):
  return [x, x**2]

p3_x = [quad_phi(x_val[0]) for x_val in d1_xvals]


log = LogisticRegression(penalty= None)
log.fit(p3_x, d1_yvals)

p3_plotx = np.linspace(-6, 2)
p3_ploty = [quad_phi(x) for x in p3_plotx]


plt.plot(p3_plotx, log.predict(p3_ploty), c = 'g')
plt.scatter(d1_xvals, d1_yvals)
plt.title('Dataset 1: Quadratic Logistic Regression')
plt.show()

#Comparison

plt.scatter(d1_xvals, d1_yvals, c = 'b')
plt.plot(p2_plotx, p2_ploty, c = 'r')
plt.plot(p3_plotx, log.predict(p3_ploty), c = 'g')
plt.title('Regression Comparison')
plt.show()




pass
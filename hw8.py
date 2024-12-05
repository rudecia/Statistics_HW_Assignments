import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
from sklearn.linear_model import LinearRegression as lr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import root_mean_squared_error

with open('/Users/rudeciabernard/Downloads/Dataset3.txt', 'r') as file:
    d3 = file.read()
    
d3 = d3.split('\n')
d3.pop()

def cleanup(s: str ):
    proto = s.split('  ')
    z = list(filter(lambda s: s != '', proto))
    z[0] = z[0].lstrip()
    return list(map(float, z))

dataset3 = list(map(cleanup, d3))

#________________________________________#
#Problem 1: Simple Linear Regression

regression_x_vals = [[p[0]] for p in dataset3]
x_vals = [p[0] for p in dataset3]

regression_y_vals = [[p[1]] for p in dataset3]
y_vals = [p[1] for p in dataset3]


linear_mod = lr()
linear_mod.fit(regression_x_vals, regression_y_vals)
linear_predictions = linear_mod.predict(regression_x_vals)

var_mle = sum([(regression_y_vals[n] - linear_predictions[n]) **2 for n in range(0, 200)])/200
unbiased_mle = var_mle * (200/198)

#alphas and betas
print(f'the intercept is for model 1 is {linear_mod.intercept_} and the coefficient is {linear_mod.coef_}.')

#variance
print(f'the biased and unbiased variances for model 1 are {var_mle} and {unbiased_mle} respectively')


#plots
plt.plot(x_vals, linear_predictions, c = 'r')
plt.scatter(x_vals, y_vals)
plt.title('Dataset 3 Simple Linear Regression')
plt.show()
#_____________________________________#
#Problem 2: Dataset 3 Modeled as a Quadratic Regression

poly = PolynomialFeatures(degree= 2)
poly_features = poly.fit_transform(np.array(x_vals).reshape(-1, 1))

poly_reg = lr() 
poly_reg.fit(poly_features, regression_y_vals)

quadratic_predictions = poly_reg.predict(poly_features)

plt.scatter(x_vals, y_vals)
#np.sort prevents it from jumping around weirdly
plt.plot(np.sort(x_vals), quadratic_predictions[np.argsort(x_vals)], c= 'r')
plt.title('Dataset 3 Quadratic Regression')
plt.show()

var2_mle = sum([(y_vals[n] - quadratic_predictions[n][0]) **2 for n in range(0, 200)])/200
var2_mle_ub = var2_mle * 200/(200 - 3)

#alphas and betas
print(f'the intercept is for model 2 is {poly_reg.intercept_} and the coefficients are {poly_reg.coef_[0][1:3]}.')
#variance
print(f'the variance estimate for model 2 is {var2_mle} while the unbiased estimate is {var2_mle_ub}')


#_______________________________________
#Problem 3: Dataset 3 Harmonic + Polynomial Regression

def phi(x, n):
    l = [x, x**2]
    for num in range(1, n):
        l.append(np.sin(x * num))
        l.append(np.cos(x * num))
    return l

mod3 = [phi(x, 20) for x in x_vals]

reg_3 = lr()
reg_3.fit(mod3, regression_y_vals)
pred_3 = reg_3.predict(mod3)

#plots
plt.plot(np.sort(x_vals), pred_3[np.argsort(x_vals)], c = 'r')
plt.scatter(x_vals, y_vals)
plt.title('Dataset 3 Harmonic + Polynomial Regression')
plt.show()

#alphas and betas
print(f'the intercept for model 3 is {reg_3.intercept_} and the first three betas are {reg_3.coef_[0][0:3]}')

#variance
var3_mle = sum([(y_vals[n] - pred_3[n][0]) **2 for n in range(0, 200)])/200
var3_mle_ub = var3_mle * 200/(200-41)

print(f'the variance estimate is {var3_mle} and the unbiased estimate is {var3_mle_ub}')




#Problem 4 Cross-Validation
x_train = x_vals[0:100]
y_train = y_vals[0:100]

x_test = x_vals[101:200]
y_test = y_vals[101:200]

def phi_truncator(x_val, dimension):
    return phi(x_val, 20)[0:dimension]

def rmse_calculation(x, y, d):
    l = []
    for d in range(1, 41):
        train_features = [phi_truncator(x_value, d) for x_value in x_vals[0:100]]
        test_features =  [phi_truncator(x_value, d) for x_value in x]
        model = lr()
        model.fit(train_features, regression_y_vals[0:100])
        l.append(root_mean_squared_error(y, model.predict(test_features)))
    return l


rmse_j = list(range(1, 41))
rmse_train = rmse_calculation(x_train, y_train, 40)
rmse_test = rmse_calculation(x_test, y_test, 40)

plt.plot(rmse_j, rmse_train, c = 'b')
plt.plot(rmse_j,rmse_test, c = 'r' )
plt.title('RMSE vs j')
plt.show()


min_train_j = rmse_train.index(min(rmse_train)) + 1 # equal to 40
min_test_j  = rmse_test.index(min(rmse_test)) + 1 #equal to 8

#4b is the same as 3b


test_mod_features = [phi_truncator(x, 8) for x in x_vals]
test_mod = lr()
test_mod.fit(test_mod_features, regression_y_vals)

plt.plot(np.sort(x_vals), test_mod.predict(test_mod_features)[np.argsort(x_vals)], c = 'r')
plt.scatter(x_vals, y_vals)
plt.title('Harmonic + Polynomial Regression (8 dimensional)')
plt.show()



#Problem 5: Extrapolation
extrapolation_nums = np.linspace(-4, 4, 100)
p5  = [[n] for n in extrapolation_nums]
p5_quad = [[1] + phi_truncator(x, 2) for x in extrapolation_nums]
p5_40d = [phi(x, 20) for x in extrapolation_nums]
p5_8d = [phi_truncator(x, 8) for x in extrapolation_nums]

#linear model
plt.plot(extrapolation_nums, linear_mod.predict(p5), c = 'r')
plt.scatter(x_vals, y_vals)
plt.title('linear on interval [-4, 4]')
plt.show()

#quadratic
plt.plot(extrapolation_nums, poly_reg.predict(p5_quad), c = 'r')
plt.scatter(x_vals, y_vals)
plt.title('quadratic on [-4, 4]')
plt.show()

#quadratic + harmonic 40d 
plt.plot(extrapolation_nums, reg_3.predict(p5_40d), c = 'r')
plt.scatter(x_vals, y_vals)
plt.title('quadratic+harmonic 40d on interval [-4, 4]')
plt.show()

#quadratic + harmonic 8d
plt.plot(extrapolation_nums, test_mod.predict(p5_8d), c= 'r')
plt.scatter(x_vals, y_vals)
plt.title('quadratic+harmonic 8d on interval[-4, 4]')
plt.show()



pass



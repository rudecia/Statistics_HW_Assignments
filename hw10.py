import random
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

random.seed(42)


def confidence_interval(s):
    contains_3 = 0
    for sample in s:
        upper_bound = np.mean(sample) + 1.96 * ((np.mean(sample)/len(sample))**0.5)
        lower_bound =  np.mean(sample) - 1.96 * ((np.mean(sample)/len(sample))**0.5)
        if 3 <= upper_bound and 3 >= lower_bound:
            contains_3 += 1
       
    return contains_3 / len(s)

#1B: Monte Carlo approximation w/ n = 5
sample_1 = np.random.poisson(3, (10**5, 5))
print(confidence_interval(sample_1))


#1C: Monte Carlo approximation w/ n = 100
sample_2 = np.random.poisson(3, (10**5, 100))
print(confidence_interval(sample_2))

#Problem 4: False Discovery Rate
a = 10
b = 10**4 - a

def phi(xk):
    return 2 * norm.cdf(- abs(xk))

all_samples = np.concatenate((np.random.normal(4, 1, (1000, a)), np.random.normal(0, 1, (1000, b))), axis = 1)
sample_p_vals = [phi(x) for x in all_samples]


test_x = np.concatenate((np.random.normal(4, 1, (3, 5)), np.random.normal(0, 1, (3, 5))), axis = 1)
test_p_vals = [phi(x) for x in test_x]




def ben(p: list):
    'returns the null hypotheses that have been rejected based on the Benjamin Hochberg procedure'
    bh = p.copy()
    bh.sort()
    rejected_nulls = []
    for index, number in enumerate(bh):
        k = index + 1
        if number <= (0.3 * k / 10**4):
            rejected_nulls.append((k, number))

    return rejected_nulls


def find_f1(p: list):
    'computes f1 and n1 for a sample'
    false_discoveries = 0
    positives = ben(p)
    if len(positives) == 0:
        return (0,0)
    else:
        for discovery in positives:
            if not(list(p).index(discovery[1]) in range(0, 10)):
                false_discoveries += 1
    return (false_discoveries/len(positives), false_discoveries)


def type_ii_error(p: list):
    'computes the number of type ii errors'
    discoveries = len(ben(p))
    num_false = find_f1(p)[1]
    true_disc = discoveries - num_false
    return 10 - true_disc



f1 = [find_f1(p)[0] for p in sample_p_vals]
ti_errors = [find_f1(p)[1] for p in sample_p_vals]
tii_errors = [type_ii_error(p) for p in sample_p_vals]
n_reject = [len(ben(p)) for p in sample_p_vals]

#find the indices of the values where there are zeros for n_reject
y = list(filter(lambda s: s[1] == 0, list(enumerate(n_reject))))
#they are 376 and 838


z = f1.copy()

z.pop(65)
z.pop(101)


#Problem 4a
print(f'find_f1(P) is {np.mean(f1)} so the Benjamin Hochberg procedure seems to be controlling error')
print(f'E(F1|N_reject > 0) is {np.mean(z)}')

#Problem 4b
plt.hist(f1)
plt.title('F1 Histogram')
plt.show()


plt.hist(ti_errors)
plt.title('NI Histogram')
plt.show()


plt.hist(tii_errors)
plt.title('N2 Histogram')
plt.show()

plt.hist(n_reject)
plt.title('N_Reject Histogram')
plt.show()


print(f'The proportion of F1 > 0.3 is {len(list(filter(lambda s: s > 0.3, f1)))/len(f1)}')


pass
import numpy as np
import math
import random
import pandas as pd
#HW 2 
#4A
def MCSample(P: list, m: list, n: int) -> list:
    'creates a sequence of length n from probability matrix P and initial probability vector m'
    sequence = []
    states = list(range(1, len(m)+1))
    x1 = random.choices(states, weights = m, k = 1)
    sequence.append(x1[0])
    for number in sequence:
        if len(sequence) == n:
            break
        else:
            sequence.append(random.choices(states, weights = P[(number - 1)], k = 1)[0])
    return sequence

X = MCSample


#4B
P = [[0.1, 0.8, .1, 0],
     [0, 0, 0.5, 0.5],
     [0, 0, 0, 1],
     [0.5, 0.2, 0.2, 0.1]]
m = [0.25, 0.25, 0.25, 0.25]

def occurence_pct(seq: list, s: int):
    'shows the fraction of times each of the 4 states appears in the sequence'
    states = list(range(1, s+1))
    return pd.DataFrame([seq.count(state)/ len(seq) for state in states], index = states, columns = ['Fraction'])

#answer
print(occurence_pct(X(P, m, 10**5), 4))

#4C

def pairs_of_states(seq: list):
    'list of the pairs to be evaluated'
    pairs = np.array_split(seq, len(seq)/2)
    return [list(map(int, list(arr))) for arr in pairs]
    
    
def pair_percentages(seq):
    'creates a dataframe with fraction of i = Xk, j =Xk+1'
    pair_list = pairs_of_states(seq) 
    unique_pairs = pd.DataFrame(pair_list).drop_duplicates().values.tolist() 
    y = pd.DataFrame([[pair[0], pair[1], pair_list.count(pair) / len(pair_list)]for pair in unique_pairs],
                      columns = ['i', 'j', 'Pij'])
    return y.pivot_table(index = 'i', columns= 'j', values= 'Pij').fillna(value = 0)

ij_table = pair_percentages(X(P, m, 10**5))

#answer:
print(ij_table)

#4D
def j_given_i(table: pd.DataFrame):
  jgi = []
  for num in range(0, 4):
    jgi.append([z /float(ij_table.iloc[num].sum()) 
                for z in table.iloc[num].values.tolist()])
  return pd.DataFrame(jgi, index = [1,2,3,4], columns = [1,2,3,4])

#answer:
print(j_given_i(ij_table))
      

pass
import pandas as pd
import numpy as np
import random
with open('/Users/rudeciabernard/Downloads/X.txt', 'r') as file:
    dna_seq = file.read()

random.seed(45)

#1A
def occurence_pct(seq: str, s: int):
    'shows the fraction of times each of the 4 states appears in the sequence'
    states = ['G','C','A','T']
    return pd.DataFrame([seq.count(state)/ len(seq) for state in states], index = states, columns = ['Fraction'])
#1B

def pairs_of_states(seq):
    'list of the pairs to be evaluated'
    seq1 = list(filter(lambda s: s!= '', seq.replace('', ' ').split(' ')))
    return [seq1[i:i + 2] for i in range(0, len(seq1), 2)]

def pair_percentages(seq):
    'creates a dataframe with fraction of i = Xk, j =Xk+1'
    pair_list = pairs_of_states(seq) 
    unique_pairs = pd.DataFrame(pair_list).drop_duplicates().values.tolist() 
    y = pd.DataFrame([[pair[0], pair[1], pair_list.count(pair) / len(pair_list)]for pair in unique_pairs],
                      columns = ['i', 'j', 'Pij'])
    return y.pivot_table(index = 'i', columns= 'j', values= 'Pij').fillna(value = 0)

def trans(num: int):
    state_trans = {1: 'A', 2: 'C', 3: 'G', 4: 'T'}
    return state_trans[num]

#1D
def MCSample_DNA(P: list, m: list, n: int) -> list:
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
    return list(map(trans, sequence))

P1 = [[0.086971, 0.050460, 0.073332, 0.067322],
     [0.074524, 0.061677, 0.012454, 0.073479],
     [0.060279, 0.049459, 0.061506, 0.050324],
     [0.056433, 0.060373, 0.074353, 0.087055]]

m1 = [0.278146,0.222051, 0.221606, 0.278197 ]

Y1 = MCSample_DNA(P1, m1, 10**5)

def ct_letters_in_row(sequence, letter: str,ct: int):
    return sequence.count(letter * ct)/len(sequence)

#letter in a row prevalence
A = pd.DataFrame([ct_letters_in_row(dna_seq, 'A', num) for num in range(3, 10)], [ct_letters_in_row(''.join(Y1), 'A', num) for num in range(3, 10)])
G = pd.DataFrame([ct_letters_in_row(dna_seq, 'G', num) for num in range(3, 10)], [ct_letters_in_row(''.join(Y1), 'G', num) for num in range(3, 10)])
C = pd.DataFrame([ct_letters_in_row(dna_seq, 'C', num) for num in range(3, 10)], [ct_letters_in_row(''.join(Y1), 'C', num) for num in range(3, 10)])
T = pd.DataFrame([ct_letters_in_row(dna_seq, 'T', num) for num in range(3, 10)], [ct_letters_in_row(''.join(Y1), 'T', num) for num in range(3, 10)])

#Question 6
#6A
def Noisy_Sequence(Z_sequence: list, C: list):
    states = list(range(1, len(C[0])+1))
    X_sequence = [random.choices(states, weights = C[(Z-1)], k =1)[0] for Z in Z_sequence]
    return X_sequence
X = Noisy_Sequence
#6B
zseq = [1, 1, 1, 2, 2, 2, 1, 2, 1, 2]
C6b= [[1, 0, 0],
    [0, 0.3, 0.7]]

print(Noisy_Sequence(zseq, C6b))

#6C
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


def HMM_Sample(Cz: list, Pz: list, mz: list, nz: int):
    Z_values = MCSample(Pz,mz, nz)
    X_values = Noisy_Sequence(Z_values, Cz)
    return [Z_values, X_values]
    
#6D
P6d = [[0.2, 0.7, 0.1],
      [0.4, 0, 0.6],
      [0,   1,  0]]

C6d = [[0.9, 0.1], 
     [0.2, 0.8], 
     [0.5, 0.5]]

m6d = [1/3, 1/3, 1/3]

n = 10**5

x1n = HMM_Sample(C6d, P6d, m6d, n)[1]

fraction = pd.DataFrame([x1n.count(1)/len(x1n), x1n.count(2)/len(x1n)], index = [1, 2])
print(fraction)

#Question 7C
P7 = [[0,   1, 0,  0,   0], 
      [0.2, 0, 0.8,0,   0],
      [0,   0,  0 ,1,   0],
      [0,   0,  0, 0,   1],
      [0.9, 0,  0.1, 0, 0]]
m7 = [0.5, 0, 0.5, 0, 0]

C7 = [[1,0],
      [1,0],
      [0,1],
      [0,1],
      [0,1]]

def Noisy_Sequence2(Z_sequence: list, C: list):
    'for partial observations'
    states = ['A', 'B']
    X_sequence = [random.choices(states, weights = C[(Z-1)], k =1)[0] for Z in Z_sequence]
    return X_sequence

print(Noisy_Sequence2(MCSample(P7, m7, 20), C7))



pass


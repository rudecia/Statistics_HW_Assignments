import pandas as pd
import numpy as np
import random
import itertools as it

P = [[0.2, 0.7, 0.1],
      [0.4, 0, 0.6],
      [0,   1,  0]]

C = [[0.9, 0.1], 
     [0.2, 0.8], 
     [0.5, 0.5]]

M = [0.3335, 0.3333, 0.3332]

N = 10

x1_10 = [2, 1, 1, 1, 2, 1, 2, 2, 1, 2]

#Q3
all_1_2_3seqs = [list(filter(lambda s: s!= '',''.join(seq).replace('', " ").split(' '))) 
                for seq in it.product('123', repeat= 10)]
all_1_2_3seqs = [list(map(int, seq)) for seq in all_1_2_3seqs]

def p_z(zk, zk1):
    return P[zk -1][zk1 - 1]

def p_z_given_x(zk, zk1, xk1):
    p_x_given_z = C[zk1-1][xk1-1]
    p_z = P[zk -1][zk1 - 1]
    denom = sum([C[n][xk1-1] * P[zk-1][n] for n in range(0, 3)])
    return (p_z * p_x_given_z) / denom 

def p_z1_given_x1_max(x1):
    denom = sum([M[n]*C[n][x1 -1] for n in range(0, 3)])
    z1_probs = [(M[m]*C[m][x1 -1]) / denom for m in range(0, 3)]
    return z1_probs.index(max(z1_probs)) + 1

def p_zseq_given_xseq(x_seq, z_sequence):
    for index, val in enumerate(z_sequence):
        product = M[z_sequence[0] - 1] * C[z_sequence[0]-1][x_seq[0] - 1]
        if index == 0:
            pass
        else:
            product = product * p_z_given_x(z_sequence[index - 1], val, x_seq[index]) * p_z(z_sequence[index-1], val)
    return product

def p_z_seq(z_sequence):
    pz1 = M[z_sequence[0] - 1]
    product = pz1
    for index, z in enumerate(z_sequence):
        if index == 0:
            pass
        else:
            product = product * p_z(z_sequence[index -1], z)
    return product


#Question 3A: the sequence composed of the most likely z at each position
# def arg_max3a(n: int):
#     'sequence w the most likely z at each position'
#     arg_list = []
#     z1 = M.index(max(M)) + 1
#     arg_list.append(z1)
#     for arg in arg_list:
#         if len(arg_list) == n:
#             break
#         else:
#             probs = [p_z(arg, n) for n in range(1,4)]
#             arg_list.append(probs.index(max(probs))+1)
#     return arg_list

def arg_max3a_helper(z_seqs, index):
    'determine the zk with the highest probability'
    chance = [0, 0, 0]
    for z_seq in z_seqs:
     chance[z_seq[index] -1] += p_z_seq(z_seq)
    return chance.index(max(chance)) + 1

def arg_max3a(z_seqs):
    z_list = []
    for n in range(0,9):
        z_list.append(arg_max3a_helper(z_seqs, n))
    return z_list
    


arg3a = arg_max3a(all_1_2_3seqs)
print(arg3a)
print(p_z_seq(arg3a))
print(p_zseq_given_xseq(x1_10, arg3a))

#Question 3B
def arg_max3B(seq_list: list):
    'returns most likely z sequence from list of all z sequences'
    highest_prob = 0
    i = 0
    for index, sequence in enumerate(seq_list):
        prob = p_z_seq(sequence)
        if prob > highest_prob:
            highest_prob = prob
            i = index 
    return seq_list[i]

arg3b = arg_max3B(all_1_2_3seqs)
print(arg3b)
print(p_z_seq(arg3b))
print(p_zseq_given_xseq(x1_10, arg3b))

#Question 3C
def arg_max3C(x_seq, k):
    'returns most likely zk given xk'
    arg_list = []
    x1 = x_seq[0]
    arg_list.append(p_z1_given_x1_max(x1))
    for index, x in enumerate(x_seq):
       if len(arg_list) == k:
           break
       else:
        if index == 0:
            pass
        else:
            probs = [p_z_given_x(arg_list[index -1], n, x) for n  in range(1, 4)]
            arg_list.append(probs.index(max(probs)) + 1)
    return arg_list

arg3c = arg_max3C(x1_10, N)
print(arg3c)
print(p_z_seq(arg3c))
print(p_zseq_given_xseq(x1_10, arg3c))

#Question 3D
def arg_max3d(x_sequence, z_seq_list, i):
    'returns most common zk  given an x sequence'
    sum = [0, 0, 0]
    arglist = [p_zseq_given_xseq(x_sequence, z_sequence) for z_sequence in z_seq_list]
    for index, z_seq in enumerate(z_seq_list):
        z_val =  z_seq[i -1]
        if z_val == 1:
            sum[0] += arglist[index]
        elif z_val ==2:
            sum[1] += arglist[index]
        elif z_val == 3:
            sum[2] += arglist[index]
    return sum.index(max(sum)) + 1

arg3d = [arg_max3d(x1_10, all_1_2_3seqs, n) for n in range(1, 11)]    
print(arg3d)    
print(p_z_seq(arg3d))
print(p_zseq_given_xseq(x1_10, arg3d))


#Question 3E
def arg_max3e(x_seq, z_seq_list):
    'most likely z sequence given x sequence'
    highest_prob = 0
    id = 0
    for index, seq in enumerate(z_seq_list):
        prob = p_zseq_given_xseq(x_seq, seq) 
        if prob > highest_prob:
            highest_prob = prob
            id = index
    return z_seq_list[id]

arg3e = arg_max3e(x1_10, all_1_2_3seqs)
print(arg3e)
print(p_z_seq(arg3e))
print(p_zseq_given_xseq(x1_10, arg3e))


pass
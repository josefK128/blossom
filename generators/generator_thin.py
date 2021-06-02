#generator_thin.py
#random nxn adjacency matrix generator - relatively few connections

import numpy as np
import random
import sys


if len(sys.argv) > 1:
    n = int(sys.argv[1])
else:
    n = 100


#start by generating random {0,1}-matrix
print(f'n = {n}')
m = np.random.randint(0,2,(n,n))

#zero the diagonal - vertex i does not have an edge with itself
for i in range(n):
    m[i][i] = 0

#diagnostics
#print(m)
#print('\n')

#make adjacency matrix symmetric - the graph should be undirected
for i in range(n):      #row
    for j in range(n):   #col
        if j < i:         #lower triangular
            #print(f'\nm[{i}][{j}] is {m[i][j]}')
            if m[i][j] == 1:
                if random.uniform(0,1) > .8:
                    #print(f'setting m[{j}][{i}] = 1')
                    m[j][i] = 1
                else:
                    m[i][j] = 0
                    m[j][i] = 0
            else:
                if m[j][i] == 1:
                    #print(f'setting m[{i}][{j}] = 1')
                    m[j][i] = 0
                    m[i][j] = 0

#weighted matrix mw
mw = m.copy()   #deep copy
for i in range(n):      #row
    for j in range(n):   #col
        if j < i:         #lower triangular
            #print(f'\nm[{i}][{j}] is {m[i][j]}')
            if mw[i][j] == 1:
                if random.uniform(0,1) > .5:
                    #print(f'setting m[{j}][{i}] = 1')
                    mw[i][j] = 2.0
                    mw[j][i] = 2.0



#diagnostics
print('\n m = ')
print(m)
print('\n mw = ')
print(mw)

#generate a {0,1}-adjacency matrix csv file 
np.savetxt("../graphN.csv", m, delimiter = ",", fmt='%d')

#generate a float-weights adjacency matrix csv file
np.savetxt("../graphN_fweights.csv", mw, delimiter = ",")



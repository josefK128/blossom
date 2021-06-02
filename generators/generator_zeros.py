#generator_zeros.py
#random nxn adjacency matrix generator of zeros

import numpy as np
import random
import sys


if len(sys.argv) > 1:
    n = int(sys.argv[1])
else:
    n = 100


#start by generating random {0,1}-matrix
print(f'n = {n}')
m = [[0 for x in range(n)] for y in range(n)] 

#weighted matrix mw
mw = m.copy()   #deep copy

#diagnostics
print('\n m = ')
print(m)
print('\n mw = ')
print(mw)

#generate a {0,1}-adjacency matrix csv file 
np.savetxt("../graphN.csv", m, delimiter = ",", fmt='%d')

#generate a float-weights adjacency matrix csv file
np.savetxt("../graphN_fweights.csv", mw, delimiter = ",")



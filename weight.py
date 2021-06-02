#weight.py 
#sum weights of adjacency matrix lower triangle - don't count directed edges
#default filename is 'graph0.csv'


from numpy import genfromtxt
import numpy as np
import sys



fname = 'graph0.csv'
if len(sys.argv) > 1:
    fname = sys.argv[1]


#create internal numpy adjacency matrix representation of matix csv-file
#display numeric representation and type
A = genfromtxt(fname, delimiter=',')
print(f'weighted adjacency matrix {fname}:')
print(A)
print(type(A))


#sum weights in lower triangular part of matrix to get total weight
weight = 0.0
n = A.shape[0]
for i in range(n):
    for j in range(n):
        if j<i:
            weight += A[i][j]

print(f'total weight of adjacency matrix {fname} is {weight}')



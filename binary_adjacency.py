#binary_adjacency.py 
#reads in weighted adjacency matrix - outputs {0,1}-valued adjacency matrix
#input and output files are .csv
#default filename is 'graphN_fweights.csv'


from numpy import genfromtxt
import numpy as np
import sys



fname = 'graphN_fweights.csv'
if len(sys.argv) > 1:
    fname = sys.argv[1]


#create internal numpy adjacency matrix representation of matix csv-file
#display numeric representation and type
W = genfromtxt(fname, delimiter=',')
print(f'weighted adjacency matrix {fname}:')
print(W)
print(type(W))


#sum weights in lower triangular part of matrix to get total weight
B = np.zeros(W.shape, dtype='int')
n = W.shape[0]
for i in range(n):
    for j in range(n):
        if W[i][j] > 0.0:
            B[i][j] = 1

print(f'\nbinary adjacency matrix:')
print(B)
print(type(B))


#write {0,1}-adjacency matrix csv file as 'graph.csv' 
print(f'\nwriting binary adjacency matrix to file ./graph.csv')
np.savetxt("graph.csv", B, delimiter = ",", fmt='%d')



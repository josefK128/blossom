#read_csv_to_ndarray.py  
#display graph plot from adjacency matrix csv-file
#default filename is 'graph0.csv'

#display graph with node labels only - no edge weights
#G = nx.from_numpy_matrix(np.array(A))
#nx.draw(G, with_labels=True)
#plot.pyplot.show()


from numpy import genfromtxt
import numpy as np
import networkx as nx
import matplotlib as plot
import sys



fname = 'graph0.csv'
if len(sys.argv) > 1:
    fname = sys.argv[1]


#create internal numpy adjacency matrix representation of matix csv-file
#display numeric representation and type
A = genfromtxt(fname, delimiter=',')
print(A)
print(type(A))

#create Dict with key-value [i][j]:m[i][j]
n = A.shape[0]
print(f'n = {n}')
for i in range(n):
   for j in range(n):
       print(f'weight A[{i}][{j}] = {A[i][j]}')


#create graph G from adjacency matrix
G = nx.from_numpy_matrix(np.array(A))
pos=nx.shell_layout(G) # pos = nx.nx_agraph.graphviz_layout(G)
labels = nx.get_edge_attributes(G,'weight')

#display graph in shell_layout with vertex-index edge-weight labels
nx.draw_networkx(G,pos)
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
plot.pyplot.show()




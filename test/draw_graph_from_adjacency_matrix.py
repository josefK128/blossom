import networkx as nx
import numpy as np
import matplotlib as plot


A = [[0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 0, 0], [0, 0, 0, 0, 0, 0]]

G = nx.from_numpy_matrix(np.array(A))
nx.draw(G, with_labels=True)
plot.pyplot.show()

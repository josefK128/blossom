#create_graph_adjacency_matrix.py

#creates a graph from methods 'add_vertex(i)' and 'add_edge(v1,v2,e)'
#and then saves the graph as an adjacency matrix in a csv-file
#add_vertex(i) adds new vertex i to the graph, and
#add_edge(v1, v2, e) adds an edge with weight e between vertices v1 and v2.


import networkx as nx
import numpy as np
import matplotlib as plot



# Add a vertex to the set of vertices and the graph
def add_vertex(v):
  global graph
  global vertices_no
  global vertices
  if v in vertices:
    print("Vertex ", v, " already exists")
  else:
    vertices_no = vertices_no + 1
    vertices.append(v)
    if vertices_no > 1:
        for vertex in graph:
            vertex.append(0)
    temp = []
    for i in range(vertices_no):
        temp.append(0)
    graph.append(temp)

# Add an edge between vertex v1 and v2 with edge weight e
def add_edge(v1, v2, e):
    global graph
    global vertices_no
    global vertices
    # Check if vertex v1 is a valid vertex
    if v1 not in vertices:
        print("Vertex ", v1, " does not exist.")
    # Check if vertex v1 is a valid vertex
    elif v2 not in vertices:
        print("Vertex ", v2, " does not exist.")
    # Since this code is not restricted to a directed or 
    # an undirected graph, an edge between v1 v2 does not
    # imply that an edge exists between v2 and v1
    else:
        index1 = vertices.index(v1)
        index2 = vertices.index(v2)
        graph[index1][index2] = e

# Print the graph
def print_graph():
  global graph
  global vertices_no
  for i in range(vertices_no):
    for j in range(vertices_no):
      if graph[i][j] != 0:
        print(vertices[i], " -> ", vertices[j], \
        " edge weight: ", graph[i][j])

# Driver code        
# stores the vertices in the graph
vertices = []
# stores the number of vertices in the graph
vertices_no = 0
graph = []
# Add vertices to the graph
add_vertex(0)
add_vertex(1)
add_vertex(2)
add_vertex(3)
add_vertex(4)
add_vertex(5)
# Add the edges between the vertices by specifying
# the from and to vertex along with the edge weights.
add_edge(0, 1, 1)
add_edge(1, 0, 1)
add_edge(2, 4, 3)
add_edge(4, 2, 3)
print_graph()
print("Adjacency Matrix A: ", graph)



#generate a csv file
np.savetxt("../graph1.csv", graph, delimiter = ",")


G = nx.from_numpy_matrix(np.array(graph))
pos=nx.spring_layout(G) # pos = nx.nx_agraph.graphviz_layout(G)
labels = nx.get_edge_attributes(G,'weight')

nx.draw_networkx(G,pos)
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
plot.pyplot.show()




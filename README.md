### __blossom README__


* [0] If not already installed, install Python >=3.8 (which should
  include package/module management tool pip). If not, install pip also.
  check install and path by the following:
  ```>python --version```
  ```>pip --version```
  see ```https://www.python.org/downloads/``` 



* [1] clone the repo https://github.com/josefK128/blossom.git   
  ```git clone https://github.com/josefK128/blossom.git``` - creates ./blossom



* [2] cd to the root directory and create a virtual environment as follows:
  Windows
  ```>py -m venv env```
  Linux and MacOS
  ```>python3 -m venv env```

  for more information on virtual environments and venv see:
  ```https://docs.python.org/3/library/venv.html```



* [3] activate the virtual environment
  ```>source env/Scripts/Activate```



* [4] install external modules (harmonious versions)
  ```>pip install -r requirements.txt```



* [5] '/graphN.csv' is a {0,1}-valued adjacency matrix for a randomly 
  generated 100-vertex undirected graph. 
  A weighted version is '/graphN_fweights.csv' 

* NOTE: any weighted adjacency matrix can be converted to the needed binary
  adjacency file by 'binary_adjacency.py' which writes to a default filename
  'graph.csv'. 

  Usage is:

  ```>py binary_adjacency.py graphN_fweights.csv```    #-> graph.csv

* NOTE: a random binary graph of any order can be generated (**although not 
  necessary**) to '/graphN.csv' and '/graphN_fweights.csv' by going to the 
  /generators directory and running one of 'generator_thin.py' 
  or 'generator_thick.py', where 'thin refers to a tendency to generate a 
  more sparse matrix - 'thick', the opposite, and N is the dimension of the 
  adjacency matrix, i.e. the number of graph vertices.
  
  Usage is:

   ```>py generator_thin.py 100```    
   ```>py generator_thick.py 100```



* [6] run the blossom algorithm on any {0,1}-valued adjacency matrix represented
  as a csv-file of zeros and ones - for example 'graphN.csv'. A weighted
  adjacency matrix csv-file, if it exists in the same directory, will also
  be read into blossom.py. If used it must follow the naming convention
  'graphN_fweights.csv' for the {0,1}-valued csv-file 'graphN.csv'.
  The module uses the weighted adjacency matrix to produce a weighted maximum
  match adjacency matrix 'graphN_max_fweights.csv'. If no weighted version is
  present the module uses the {0,1}-valued adjacency matrix values as weights.
  
  The module also reports the number of vertex matches in a maximum matching 
  (counting both vertices of match edges), and also generates an array of 
  edge tuples (both directions) for the maximum matching, corresponding to the 
  given adjacency matrix - for example 'graphN.csv'.

  Usage is:
  ```>py blossom.py  graphN.csv```

  The module creates two files:
  graphN_max_fweights.csv - the weighted adjacency matrix for the maximum match
  graphN_max_edges.txt - an array of tuples [(v1,v2),...] of the match edges



* [7] calculate the total weight of the weighted adjacency matrix
  ```>py weight.py  graphN_max_fweights.csv```



* [8] display any graph corresponding to a csv-file of its adjacency matrix (or weighted adjacency matrix). For example:
  ```>py display.py  graphN.csv```
  ```>py display.py  graphN_fweights.csv```
  ```>py display.py  graphN_max_fweights.csv```



* [9] display the edges of the maximum match. For example:
  ```>py display_max_edges.py  graphN_max_edges.txt```

  

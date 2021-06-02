### __blossom README__


* [1] clone the repo https://github.com/josefK128/blossom.git   
  ```git clone https://github.com/josefK128/blossom.git``` - creates ./blossom


* [2] cd to the root directory and activate the virtual environment - currently runs Python 3.9.1 
  ```>source env/Scripts/Activate```


* [3] install external modules  
  ```>pip install -r requirements.txt```

  
* [4] '/graphN.csv' is a {0,1}-valued adjacency matrix for a randomly generated 100-vertex undirected graph. Its weighted version is '/graphN_fweights.csv'


   NOTE: a random graph of any order can be generated (**although not necessary**) 
   to '/graphN.csv' and '/graphN_fweights.csv' by going to the /generators 
   directory and running one of 'generator_thin.py' or 'generator_thick.py', 
   where 'thin refers to a tendency to generate a more sparse matrix - 'thick',
   the opposite, and N below is the dimensions of the adjacency matrix, i.e. the
   number of graph vertice - for example:

   ```>py generator_thin.py 100```    
   ```>py generator_thick.py 100```

   
* [5] run the blossom algorithm - it reports the number of matches in a maximum matching, and also generates a list of edges (both directions) for the maximum matching, for the graph corresponding to the given adjacency matrix. The optional 2nd argument is a filename in which to write the edges of the maximum match.
  ```>py blossom.py  graphN.csv [graphN_maximum.txt]```


* [6] calculate the total weight of the weighted adjacency matrix
  ```>py weight.py  graphN_fweights.csv```


* [7] display any graph corresponding to a csv-file of its adjacency matrix (or weighted adjacency matrix). For example:
  ```>py display.py  graphN_fweights.csv```
  

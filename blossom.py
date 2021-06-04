#blossom.py 
#
#  run the blossom algorithm on any {0,1}-valued adjacency matrix represented
#  as a csv-file of zeros and ones - for example 'graphN.csv'. A weighted
#  adjacency matrix csv-file, if it exists in the same directory, will also
#  be read into blossom.py. If used it must follow the naming convention
#  'graphN_fweights.csv' for the {0,1}-valued csv-file 'graphN.csv'.
#  The module uses the weighted adjacency matrix to produce a weighted maximum
#  match adjacency matrix 'graphN_max_fweights.csv'. If no weighted version is
#  present the module uses the {0,1}-valued adjacency matrix values as weights.
#  
#  The module also reports the number of vertex matches in a maximum matching 
#  (counting both vertices of match edges), and also generates an array of 
#  edge tuples (both directions) for the maximum matching, corresponding to the 
#  given adjacency matrix - for example 'graphN.csv'.
#
#  Usage is:
#  ```>py blossom.py  graphN.csv```
#
#  The module creates two files:
#  graphN_max_fweights.csv - the weighted adjacency matrix for the maximum match
#  graphN_max_edges.txt - an array of tuples [(v1,v2),...] of the match edges
#
#
#  program entry is at main() - line 427
#  there are 3 central blocks in main:
#
#  @A - read or prepare a weights-matrix W for use in @C below
#       read in weights file (exp graphN_fweights.csv) - store as matrix W
#
#  @B - find the maximum match graph - report number of nodes
#
#  @C - write match edges (list of list-pairs) and weighted adjacency matrix
#       exp: graphN_max_edges.txt  and  graphN_max_fweights.csv





from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

from builtins import range
import re,sys,csv
import os.path

import logging
from past.utils import old_div
from builtins import object

#key modules
import numpy as np
import networkx as nx
import matplotlib as plot



# Logger setup - output errors only to blossom_errorfile
logger = logging.getLogger(__name__)
f_handler = logging.FileHandler("blossom_errorfile.log")
f_handler.setLevel(logging.ERROR)
f_format = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
f_handler.setFormatter(f_format)
logger.addHandler(f_handler)



USAGE = "Usage: {} 'infile.csv' ['outfile.txt']".format(sys.argv[0])

args_pattern = re.compile(
    r"""
    ^
    (
    (?P<HELP>-h|--help)|
    ((?P<ARG1>\w+\.csv))
    (\s(?P<ARG2>\w+\.txt$))?
    )
    $
""",
    re.VERBOSE,
)



def parse(arg_line):
    args = {}
    match_object = args_pattern.match(arg_line)
    if match_object:
        args = {
            k: v
            for k, v in list(match_object.groupdict().items())
            if v is not None
        }
    return args


def read_infile(infile):
    node_array = []
    with open(infile) as csvfile:
        for row in csv.reader(csvfile, delimiter=str(",")):
            neighbours = [idx for idx, row in enumerate(row) if row == "1"]
            node_array.append(neighbours)
    if len(node_array) == 0:
        raise SystemExit("Empty graph. Please supply a valid graph.")
    return node_array


def compute_max_matching(node_array):
    # Create node instances, fill node neighbours
    nodelist = [Node() for _ in range(len(node_array))]
    for idx, node in enumerate(node_array):
        nodelist[idx].neighbors = [nodelist[node] for node in node]

    # Create graph instance, construct graph
    graph = Graph()
    graph.nodes = {node.name: node for node in nodelist}
    graph.compute_edges()

    # Compute maximum matching
    graph.find_max_matching()

    return graph



#def save_matched_pairs(matched_dict, outfile):
#    with open(outfile, "w") as textfile:
#        for pair in list(matched_dict.items()):
#            #string = "{}:{}\n".format(pair[0], pair[1])
#            string = f"{pair[0]},{pair[1]}\n"
#            textfile.write(string)

def save_matched_pairs(matched_dict, outfile, W, wname):
    #save match edges as [v1,v2] pairs of vertex indices
    #match edge at [v1,v2] => match edge at [v2,v1] also since graph undirected 
    with open(outfile, "w") as textfile:
        for pair in list(matched_dict.items()):
            #string = "{}:{}\n".format(pair[0], pair[1])
            string = f"{pair[0]},{pair[1]}\n"
            textfile.write(string)

    #save match weights as csv-file 'wname' - weighted match adjacency matrix
    m = np.zeros(W.shape, dtype='float')
    for pair in list(matched_dict.items()):
        #print(f'pair[0] = {pair[0]} pair[1] = {pair[1]}')
        #print(f'weight = {W[pair[0]][pair[1]]}')
        m[pair[0],pair[1]] = W[pair[0],pair[1]]
    np.savetxt(wname, m, delimiter=',')





class Node(object):
    _name = 0

    def __init__(self):
        self.name = Node._name
        Node._name += 1
        self.neighbors = []
        self.match = None
        self.mark = False
        self.parent = None
        self.root = None

    def ancestor_path(self):
        """Compute the path from a node to its root"""
        path = [self]
        node = self
        while node != node.root:
            node = node.parent
            path.append(node)
        return path

    def cycle_aug_path(self, match_node, cycle):
        """Compute an augmenting path on a cycle graph"""
        idx1 = cycle.index(self)
        idx2 = cycle.index(match_node)
        path = []

        if (idx1 > 0 and idx2 == idx1 - 1) or (
            idx1 == 0 and idx2 == len(cycle) - 1
        ):
            reverse_cycle = cycle[idx1::-1] + cycle[:idx1:-1]
            for node in reverse_cycle:
                path.append(node)
                if node.match not in reverse_cycle:
                    return path
            logger.error("No matched node in reverse_cycle.")

        else:
            forward_cycle = cycle[idx1::] + cycle[:idx1]
            for node in forward_cycle:
                path.append(node)
                if node.match not in forward_cycle:
                    return path
            logger.error("No matched node in forward_cycle.")


class Supernode(Node):
    def __init__(self, cycle=None):
        super(Supernode, self).__init__()
        self.cycle = cycle

    def contract_nodelist(self, nodelist):
        """Contract a cycle to a supernode"""
        # Remove cycle from nodelist
        nodelist = [node for node in nodelist if node not in self.cycle]
        nodelist.append(self)

        # Compute supernode neighbors
        for node in self.cycle:
            if node.match and node.match not in self.cycle:
                self.match = node.match
            for neighbor in node.neighbors:
                if neighbor not in self.cycle:
                    self.neighbors.append(neighbor)
        self.neighbors = list(set(self.neighbors))

        # Modify node neighbors if neighbor in cycle
        for node in nodelist:
            if node.match in self.cycle:
                node.match = self
            node.neighbors = [
                neighbor
                for neighbor in node.neighbors
                if neighbor not in self.cycle
            ]
            if node in self.neighbors:
                node.neighbors.append(self)

        return nodelist

    def expand_nodelist(self, nodelist):
        """Expand a supernode to a cycle"""
        # Remove supernode from nodelist
        nodelist = [node for node in nodelist if node is not self]
        for node in nodelist:
            node.neighbors = [
                neighbor for neighbor in node.neighbors if neighbor is not self
            ]

        # Modify node neighbors if node is cycle neighbor and not in cycle
        for cnode in self.cycle:
            nodelist.append(cnode)
            if cnode.match and cnode.match not in self.cycle:
                cnode.match.match = cnode
            for node in cnode.neighbors:
                if node not in self.cycle:
                    node.neighbors.append(cnode)

        return nodelist

    def expand_path(self, path, cycle):
        """Replace supernode in augmenting path with corresponding cycle
        nodes"""
        if self not in path:
            return path

        elif self == path[0]:
            for node in cycle:
                if path[1] in node.neighbors:
                    if node.match:
                        cpath = node.cycle_aug_path(node.match, cycle)
                    else:
                        cpath = [node]
                    return cpath[::-1] + path[1:]
            logger.error("Supernode (head) not connected to rest of graph.")

        elif self == path[-1]:
            for node in cycle:
                if path[-2] in node.neighbors:
                    if node.match:
                        cpath = node.cycle_aug_path(node.match, cycle)
                    else:
                        cpath = [node]
                    return path[:-1] + cpath
            logger.error("Supernode (tail) not connected to rest of graph.")

        else:
            idx = path.index(self)
            if path.index(self.match) == idx - 1:
                for node in cycle:
                    if path[idx + 1] in node.neighbors:
                        cpath = node.cycle_aug_path(node.match, cycle)
                        return path[:idx] + cpath[::-1] + path[idx + 1 :]
                logger.error(
                    "Supernode (inner 1) not connected to rest of graph."
                )

            elif path.index(self.match) == idx + 1:
                for node in cycle:
                    if path[idx - 1] in node.neighbors:
                        cpath = node.cycle_aug_path(node.match, cycle)
                        return path[:idx] + cpath + path[idx + 1 :]
                logger.error(
                    "Supernode (inner 2) not connected to rest of graph."
                )

            else:
                logger.error(
                    "Supernode not matched correctly to rest of graph."
                )


class Graph(object):
    def __init__(self):
        self.nodes = None
        self.edges = None


    def nvertices(self):
        return len(self.nodes)

    def nedges(self):
        return len(self.edges)


    def compute_edges(self):
        self.edges = {}
        for key in self.nodes:
            for node in self.nodes[key].neighbors:
                self.edges[tuple(sorted([key, node.name]))] = 1

    def mark_edges(self, node1, node2):
        self.edges[tuple(sorted([node1.name, node2.name]))] = 0

    def clean_graph(self):
        for key in self.nodes:
            self.nodes[key].mark = False
            self.nodes[key].parent = None
            self.nodes[key].root = None

    def compute_size_matching(self):
        """Compute number of matched pairs"""
        size = 0
        for key in self.nodes:
            if self.nodes[key].match:
                size += 1
        assert size % 2 == 0
        return old_div(size, 2)

    def create_matching_dict(self):
        """Create dictionary of matched pairs"""
        matching_dict = {}
        for key in self.nodes:
            if self.nodes[key].match:
                matching_dict[key] = self.nodes[key].match.name
        return matching_dict

    def find_max_matching(self):
        """Wrapper function for computing maximum matching"""
        path = self.find_aug_path()
        if not path:
            return self
        else:
            self.aug_old_matching(path)
            return self.find_max_matching()

    def find_aug_path(self):
        """Edmonds algorithm for computing maximum matching"""
        self.clean_graph()
        self.compute_edges()

        exposed_node = [
            node for node in list(self.nodes.values()) if node.match is None
        ]
        for node in exposed_node:
            node.parent = node
            node.root = node

        for node in exposed_node:
            if not node.mark:
                for adj_node in node.neighbors:
                    if self.edges[tuple(sorted([node.name, adj_node.name]))]:
                        if adj_node not in exposed_node:
                            adj_node.parent = node
                            adj_node.root = node.root
                            adj_node.mark = True  # odd distance from root
                            self.mark_edges(node, adj_node)
                            exposed_node.append(adj_node)

                            adj_match = adj_node.match
                            adj_match.parent = adj_node
                            adj_match.root = adj_node.root
                            self.mark_edges(adj_node, adj_match)
                            exposed_node.append(adj_match)
                        else:
                            if not (len(adj_node.ancestor_path()) % 2):
                                self.mark_edges(node, adj_node)
                            else:
                                if node.root != adj_node.root:
                                    path1 = node.ancestor_path()
                                    path2 = adj_node.ancestor_path()
                                    return path1[::-1] + path2
                                else:
                                    return self.blossom(node, adj_node)
                node.mark = True

        return []

    def blossom(self, node1, node2):
        """Find augmenting path on blossom (cycle)"""
        path1 = node1.ancestor_path()
        path2 = node2.ancestor_path()
        cycle = path1[::-1] + path2[:-1]

        # Contract cycle nodes to supernode
        snode = Supernode(cycle)
        nodelist = snode.contract_nodelist(list(self.nodes.values()))
        self.nodes = {node.name: node for node in nodelist}
        self.compute_edges()
        aug_path = self.find_aug_path()

        # Expand supernode back to original cycle nodes
        aug_path = snode.expand_path(aug_path, cycle)
        nodelist = snode.expand_nodelist(list(self.nodes.values()))
        self.nodes = {node.name: node for node in nodelist}
        self.compute_edges()

        return aug_path

    @staticmethod
    def aug_old_matching(path):
        """Apply augmenting path to current matching on graph"""
        for idx, node in enumerate(path):
            if (idx + 1) % 2:
                node.match = path[idx + 1]
            else:
                node.match = path[idx - 1]




def main():
    args = parse(" ".join(sys.argv[1:]))
    if not args:
        raise SystemExit(USAGE)
    if args.get("HELP"):
        print(USAGE)
        return

    #print(f'args = {args}')


    #@A - read or prepare a weights-matrix W for use in @C below
    #read in weights file (exp graphN_fweights.csv) - store as matrix W
    stem = sys.argv[1].split('.')
    #print(f'stem[0] = {stem[0]}')
    #print(f'stem[1] = {stem[1]}')
    wname = stem[0] + '_fweights.' + stem[1]

    #if the weighted adjacency file exists use it
    #otherwise use the {0,1}-weighted adjacency martrix
    if os.path.exists(wname) == False:
        wname = sys.argv[1]
        print(f'weights-file does not exist so use {0,1}-weights in {wname}:')
        wname = sys.argv[1]
    else:
        print(f'weights file exists with wname = {wname}:')

    #store the weights in a 2-dim array corresponding to the adjacency matrix
    #W is type numpy.ndarray
    W = np.genfromtxt(wname, delimiter=',')
    print(W)
    #print(type(W))



    #@B - find the maximum match graph - report number of nodes
    #     node_array is a list of indices of edge-pairs occurring in each row.
    #     For exp. [[9], [], [1,10],...] => edges at (0,9),(2,1),(2,10),etc...
    #     matched_graph is an instance of class __main__.Graph (defined line309)
    node_array = read_infile(args["ARG1"])
    #print(f'node_array = {node_array}')
    #print(f'type(node_array) = {type(node_array)}')

    #find matching graph
    matched_graph = compute_max_matching(node_array)
    #print(f'type(matched_graph) = {type(matched_graph)}')

    # Multiple by two to convert number of matched pairs to matched nodes.
    outstring = (
        """\nThere are {} matched nodes in maximum matched graph.""".format(
            int(2 * matched_graph.compute_size_matching())
        )
    )
    print(outstring)



    #@C - write match edges (list of list-pairs) and weighted adjacency matrix
    #     exp: graphN_max_edges.txt  and  graphN_max_fweights.csv
    #     matched_dict is a Dict v1:v2 of edge vertex pairs
    #     NOTE: v1:v2 in matched_dict => v2:v1 is also in matched_dict
    max_edges_fname = args.get("ARG2")
    #print(f'max_edges_fname = {max_edges_fname}')
    if max_edges_fname is None:
        max_edges_fname = stem[0] + '_max_edges.txt' #doesn't exist - create
    #print(f'max_edges_fname = {max_edges_fname}')

    matched_dict = matched_graph.create_matching_dict()
    print(f'matched edge vertex pairs are: {matched_dict}')
    #print(f'type(matched_dict) = {type(matched_dict)}')

    #save match edges as pairs in max_edges_fname
    #save match weights as csv-file of weighted match adjacency matrix 
    match_weights_fname = stem[0] + '_max_fweights.csv'
    #print(f'match_weights_fname = {match_weights_fname}')
    save_matched_pairs(matched_dict, max_edges_fname, W, match_weights_fname)








if __name__ == "__main__":
    print('blossom.py starting...')
    main()

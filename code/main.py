import os
import networkx as nx
import matplotlib.pyplot as plt
from scipy import linalg as la
from scipy.sparse import linalg as sla
import scipy.sparse as sparse
import scipy

from sklearn.cluster import KMeans
import numpy as np


class Solver:
    def __init__(self, G, nVertices, nEdges, k):
        self.G = G
        self.adj = self.compute_adjacency()
        self.nVertices = nVertices
        self.nEdges = nEdges
        self.k = k

    def algo1(self):
        # draw_graph(G)
        L = self.compute_laplacian()
        eValues, eVectors = self.compute_eigen2(L)

        eVectors = np.real(eVectors)
        eValues = np.real(eValues)

        # evectors is U
        C = self.kmean(eVectors)

        # TODO:
        # For each line, return the associated cluster
        communities = C
        nodes = np.array(self.G.nodes())
        return np.stack((nodes, communities))

    def dumpOutput(self, algoName, output):
        fp = os.path.join("..", "results", algoName+".txt")
        with open(fp, "w") as f:
            f.write("# {} {} {} {}\n".format(self.G.name, self.nVertices,
                                           self.nEdges, self.k))
            for out in output.T:
                nodeID = out[0]
                community = out[1]
                f.write("{} {}\n".format(nodeID, community))


    def compute_adjacency(self):
        adj = nx.adjacency_matrix(self.G)
        return adj

    # Compute unormalized laplacian (L=D-A)
    def compute_laplacian(self):
        return nx.laplacian_matrix(self.G)

    def compute_normalized_laplacian(self):
        return nx.normalized_laplacian_matrix(self.G)

    def compute_eigen(self, M):
        # TODO: code of Antoine
        pass

    # TODO: test this, should be more robust and faster
    def compute_eigen2(self, M):
        M = sparse.csr_matrix(M.astype(float))
        return sla.eigsh(M, self.k, sigma=0, which='LM')

    # Wrong, sorted in decreasing order
    def compute_eigen3(self, M):
        M = sparse.csr_matrix(M.astype(float))
        return sla.eigs(M.astype(float), k=self.k)

    def kmean(self, M):
        if np.iscomplex(M).all():
           raise ValueError("[Solver.kmean] Expecting real Matrix, "
                            "got complex")
        else:
            M = np.real(M)

        kmeans = KMeans(n_clusters=self.k, init='k-means++', max_iter=300,
                        n_init=10, random_state=0)
        kmeans.fit(M)
        return kmeans.labels_


def import_graph(graphName):
    fp = os.path.join("..", "graphs_processed", graphName + ".txt")

    G = nx.Graph(name=graphName)
    # with open(fp) as f:
    edges = nx.read_edgelist(fp, comments="#", encoding="utf-8", nodetype=int)
    # nodes = nx.read_adjlist("nodes.txt")
    # my_graph.add_nodes_from(nodes)
    G.add_edges_from(edges.edges())

    with open(fp) as f:
        firstLine = f.readline()
        fLineSplit = firstLine.split(" ")
        nVertices = int(fLineSplit[2])
        nEdges = int(fLineSplit[3])
        k = int(fLineSplit[4])
    return G, nVertices, nEdges, k

def draw_graph(G):

    # nx.draw(G, pos, with_labels=False, font_weight='bold')
    # labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
    plt.show()
    # plt.savefig("test.png")


if __name__ =="__main__":
    print("Hello there, general Kenobi")
    graphName = "ca-AstroPh"
    G, nVertices, nEdges, k = import_graph(graphName)



    solver = Solver(G, nVertices, nEdges, k)
    output = solver.algo1()
    solver.dumpOutput(graphName, output)












# >>> G.nodes()
# ['a', 1, 2, 3, 'spam', 'm', 'p', 's']
# >>> G.edges()
# [(1, 2), (1, 3)]
# >>> G.neighbors(1)
# [2, 3]
# G.edges_iter()
# >> G[1][3]['color']='blue'
# https://networkx.github.io/documentation/networkx-1.10/tutorial/tutorial.html
# >>> nx.draw(G)
# >>> nx.draw_random(G)
# >>> nx.draw_circular(G)
# >>> nx.draw_spectral(G)
# >>> plt.show()
# >>> nx.draw(G)
# >>> plt.savefig("path.png")

# If Graphviz and PyGraphviz, or pydot, are available on your system, you can also use
# >>> nx.draw_graphviz(G)
# >>> nx.write_dot(G,'file.dot')

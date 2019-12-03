import os, sys
import networkx as nx
from utils import import_graph, draw_graph, iprint, dprint
from evaluator import Evaluator
import scipy
from scipy import linalg as la
from scipy.sparse import linalg as sla
import scipy.sparse as sparse
import pandas as pd

from sklearn.cluster import KMeans
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

"""Global variables used for information and debug msges prints"""
VERBOSE = True
DEBUG = False


class Solver:
    """
    A class to solve a graph problem as introduced in the statement
    """
    def __init__(self, G, nVertices, nEdges, k):
        self.G = G
        # TODO: following not even required:
        self.adj = self.compute_adjacency()
        self.nVertices = nVertices
        self.nEdges = nEdges
        self.k = k

    def partition(self, params):
        """
        Implementation of algo 1 for Spectral Analysis: unormalized spectral
        clustering.
        """
        # Compute unormalized laplacian
        iprint("Computing laplacian of type {}...".format(params["L"]))

        L = self.compute_laplacian(params["L"])

        # Compute the eigenvalues and corresponding eigenvectors
        iprint("Computing eigens ...")
        eValues, eVectors = self.compute_eigen3(L)

        dprint("evalues = {}".format(eValues))

        # K-mean clustering on the eigenvectors matrix
        iprint("Performing kmean ...")
        clusters = self.kmean(eVectors)
        dprint("Computed labels: {}".format(np.array(clusters)))

        # For each line, return the associated cluster
        nodes = np.array(self.G.nodes())
        return np.stack((nodes, clusters))


    def compute_adjacency(self):
        """
        Compute the adjacency matrix of the graph.
        """
        adj = nx.adjacency_matrix(self.G)
        return adj


    def compute_laplacian(self, type):
        """
        Compute the Laplacian L
        src code of library: https://networkx.github.io/documentation/networkx-1.10/_modules/networkx/linalg/laplacianmatrix.html
        :param normalize: if true: L = L=I-D^{-1/2}AD^{-1/2}, otherwise L = D-A
        :return: L
        """
        if type == "normalized":
            return nx.normalized_laplacian_matrix(self.G)
        elif type == "unormalized":
            return nx.laplacian_matrix(self.G)
        elif type == "normalized_random_walk":
            return nx.directed_laplacian_matrix(G,walk_type="random",
                                               alpha=0.95)
        else:
            raise ValueError("[compute_laplacian] Unknown type {} "
                             "provided".format(type))


    def compute_eigen(self, M):
        """
        Compute eigenvectors of the given matrix M.
        """
        M = sparse.csr_matrix(M.astype(float))
        eValues, eVectors = sla.eigs(M, k=self.k, which='SM',
                              return_eigenvectors=True)
        # which = "SM" (smallest magnitude)
        # which = "SR" (smallest real part)

        if not np.isreal(eValues).all or not np.isreal(eVectors).all:
            raise ValueError("[compute_eigen] computed complex eigen while "
                             "expecting real ones.")
        return np.real(eValues), np.real(eVectors)


    # does not work well
    def compute_eigen2(self, M):
        M = sparse.csr_matrix(M.astype(float))
        eValues, eVectors = sla.eigsh(M, self.k, sigma=0, which='LM')



    def compute_eigen3(self, M):
        M = sparse.csr_matrix(M.astype(float))
        eValues, eVectors = sla.eigsh(M, k=self.k, which='SM', v0=(np.zeros(
            self.nVertices) + np.finfo(np.float64).eps))
        if not np.isreal(eValues).all or not np.isreal(eVectors).all:
            raise ValueError("[compute_eigen] computed complex eigen while "
                             "expecting real ones.")
        return np.real(eValues), np.real(eVectors)



    def kmean(self, M):
        """
        K-means prediction on given matrix.
        """
        # Check that values are real
        if np.iscomplex(M).all():
           raise ValueError("[Solver.kmean] Expecting real Matrix, "
                            "got complex")
        else:
            M = np.real(M)

        # Apply k-means prediction
        kmeans = KMeans(n_clusters=self.k, init='k-means++', max_iter=400,
                        n_init=15, random_state=0).fit(M)

        # Get associated labels of predicted clusters
        clusters = kmeans.labels_
        return clusters

    def dumpOutput(self, algoName, output):

        fp = os.path.join("..", "results", algoName + ".txt")
        iprint("Dumping output of {} to {} ...".format(algoName, fp))
        with open(fp, "w") as f:
            f.write("# {} {} {} {}\n".format(self.G.name, self.nVertices,
                                             self.nEdges, self.k))
            for out in output.T:
                nodeID = out[0]
                cluster = out[1]
                f.write("{} {}\n".format(nodeID, cluster))



if __name__ =="__main__":

    # Import graph from txt file and create solver object
    graphName = "ca-AstroPh"
    G, nVertices, nEdges, k = import_graph(graphName)

    # TODO: grid search for best algo and best parameters (laplacian norm or
    #  not, kmean or gmm, ...)
    gridParams = [{"L": "unormalized"}, {"L": "normalized"}]


    solver = Solver(G, nVertices, nEdges, k)


    # Simple test of a single algorithm
    # ---------------------------------
    params = {"L":"unormalized"}
    try:
        output = solver.partition(params)
    except Exception as e:
        sys.exit(e)
    solver.dumpOutput(graphName, output)

    evaluator = Evaluator(solver)
    metrics = evaluator.evaluate(output)
    print(metrics)
    # evaluator.gridSearch(dumpOutputBest=True)









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

import os, sys
import networkx as nx # management of graph
from scipy.sparse import linalg as sla
import scipy.sparse as sparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
import pickle as pkl
from argparse import ArgumentParser

# Customer classes and functions
from evaluator import Evaluator
from utils import import_graph, draw_graph, iprint, dprint

np.set_printoptions(threshold=sys.maxsize)

"""Global variables used for information and debug msges prints"""
VERBOSE = True
DEBUG = False
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

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

    def make_clusters(self, params):
        """
        Perform spectral clustering.
        """
        # Compute unormalized laplacian
        iprint("Computing laplacian of type {}...".format(params["L"]))

        L = self.compute_laplacian(params["L"])

        # Compute the eigenvalues and corresponding eigenvectors
        iprint("Computing eigens ...")
        eValues, eVectors = self.compute_eigen(L, params["eigen_norm"])

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
        :param type:
            - normalized: L = L=I-D^{-1/2}AD^{-1/2}
            - unormalized: L = D-A
            - normalized_random_walk: I-D^-1A
        :return: L according to the type
        """
        # [Ulrike von Luxburg] corresponds to "L_sym"
        if type == "normalized":
            return nx.normalized_laplacian_matrix(self.G)
        elif type == "unormalized":
            return nx.laplacian_matrix(self.G)
        # [Ulrike von Luxburg] corresponds to "L_rw"
        elif type == "normalized_random_walk":
            degreeNodes = np.array(self.adj.sum(axis=0))[0]

            # Let's reverse the diagonal matrix (equivalent to invert its
            # diagonal elements)
            invDegreeNodes = 1 / degreeNodes
            i = np.arange(self.nVertices)
            invD = sparse.csr_matrix((invDegreeNodes, (i, i)),
                                     shape=self.adj.shape)

            I = sparse.identity(n=self.nVertices)
            return I - invD * self.adj
        else:
            raise ValueError("[compute_laplacian] Unknown type {} "
                             "provided".format(type))

    def _loadEigenPickle(self):
        try:

            fp1 = os.path.join(DIR_PATH, "pickle", self.G.name + "-" +
                               "eValues.pkl")
            fp2 = os.path.join(DIR_PATH, "pickle", self.G.name + "-" +
                               "eVectors.pkl")

            with open(fp1, 'rb') as fo1:
                eValues = pkl.load(fo1)

            with open(fp2, 'rb') as fo2:
                eVectors = pkl.load(fo2)

            return eValues, eVectors
        except Exception as e:
            return None

    def _saveEigenPickle(self, eValues, eVectors):
        dp = os.path.join(DIR_PATH, "pickle")
        fp1 = os.path.join(DIR_PATH, "pickle", self.G.name + "-" +
                           "eValues.pkl")
        fp2 = os.path.join(DIR_PATH, "pickle", self.G.name + "-" +
                           "eVectors.pkl")
        if not os.path.exists(dp):
            os.mkdir(dp)
        with open(fp1, 'wb') as fo1:
            pkl.dump(eValues, fo1)
        with open(fp2, "wb") as fo2:
            pkl.dump(eVectors, fo2)


    def compute_eigen(self, M, normalization=None):
        """
        Compute the eigen values and eigen vectors
        :param M: sparse 2x2 matrix
        :param normalization: type of normalization (None, l1 or l2)
        :return: real eValues, eVectors
        """
        eigens = self._loadEigenPickle()
        if eigens is None:

            M = sparse.csr_matrix(M.astype(float))
            # sla.eigs

            eValues, eVectors = sla.eigsh(M, k=self.k, which='SM', v0=(np.zeros(self.nVertices) + np.finfo(np.float64).eps))
            if not np.isreal(eValues).all or not np.isreal(eVectors).all:
                raise ValueError("[compute_eigen] computed complex eigen while "
                                 "expecting real ones.")
            if normalize is not None:
                eVectors = self._normalize(eVectors, normalization)

            eValues = np.real(eValues)
            eVectors = np.real(eVectors)

            # save pickle:
            self._saveEigenPickle(eValues, eVectors)
            return eValues, eVectors
        else:
            return eigens[0], eigens[1]


    def _normalize(self, M, type):
        """
        Generic method to normalize a matrix M
        """
        if type == "l1":
            # L1 norm (sum of rows = 1)
            return normalize(M, axis=1, norm='l1')
        elif type == "l2":
            norm = np.sqrt((M * M).sum(axis=1))
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    M[i, j] /= norm[i]
            return M
        elif type == "degree":
            # TODO
            pass
        elif type is None:
            return M
        else:
            raise ValueError("unknown norm type")


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
        kmeans = KMeans(n_clusters=self.k, init='k-means++', max_iter=300,
                        n_init=15, random_state=0, n_jobs=-1).fit(M)

        # Get associated labels of predicted clusters
        clusters = kmeans.labels_
        return clusters

    def dumpOutput(self, algoName, output):
        """
        Dump output to filesystem (/results)
        """
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

    # parsing command line arguments
    parser = ArgumentParser(description='Enter the graph you want to analyze')
    parser.add_argument('--graphName', help='The name of the graph or "all" '
                                            'if all should be analyzed',
                        default="ca-AstroPh")
    args = parser.parse_args()


    graphName = args.graphName


    graphNames = []
    if graphName != "all":
        graphNames.append(graphName)
    else:
        # graphNames = ["ca-AstroPh", "ca-CondMat", "ca-GrQc", "ca-HepPh",
        #               "ca-HepTh", "Oregon-1", "roadNet-CA", "soc-Epinions1",
        #               "web-NotreDame"]
        # graphNames = ["soc-Epinions1",
        #               "web-NotreDame"]
        graphNames = ["ca-GrQc", "Oregon-1", "soc-Epinions1",
                      "web-NotreDame", "roadNet-CA"]

    for graphName in graphNames:

        # Import graph from txt file and create solver object
        G, nVertices, nEdges, k = import_graph(graphName)

        # Instanciate a solver for a given problem instance
        solver = Solver(G, nVertices, nEdges, k)


        # # Simple test of a single algorithm
        # # ---------------------------------
        # params = {"L":"unormalized"}
        # try:
        #     output = solver.make_clusters(params)
        # except Exception as e:
        #     sys.exit(e)
        # solver.dumpOutput(graphName, output)
        #
        # evaluator = Evaluator(solver)
        # metrics = evaluator.evaluate(output)
        # print(metrics)

        # Grid search on graphName
        # ------------------------
        gridParams = [
            {"L": "unormalized", "eigen_norm": None},
            {"L": "unormalized", "eigen_norm": "l1"},
            # {"L": "unormalized", "eigen_norm": "l2"},
            #
            # {"L": "normalized", "eigen_norm": None},
            # {"L": "normalized", "eigen_norm": "l1"},
            # {"L": "normalized", "eigen_norm": "l2"},
            #
            # {"L": "normalized_random_walk", "eigen_norm": None},
            # {"L": "normalized_random_walk", "eigen_norm": "l1"},
            # {"L": "normalized_random_walk", "eigen_norm":"l2"},

        ]
        try:
            evaluator = Evaluator(solver)
            bestParams, bestMetrics, bestOutput = evaluator.gridSearch(
                gridParams, dumpOutputBest=True, plots=["score",
                                                        "n_ratio_cut",
                                                        "box_plot"])
        except Exception as e:
            sys.exit(e)
import scipy
import networkx as nx
import numpy.linalg

n = 100  # 1000 nodes
m = 500  # 5000 edges
G = nx.gnm_random_graph(n, m)

L = nx.normalized_laplacian_matrix(G)
e = numpy.linalg.eigvals(L.A)
print("Largest eigenvalue:", max(e))
print("Smallest eigenvalue:", min(e))

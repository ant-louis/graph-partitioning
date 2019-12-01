import os
import networkx as nx
import matplotlib.pyplot as plt

def create_graph(graphName):
    fp = os.path.join("..", "graphs_processed", graphName + ".txt")


    G = nx.Graph(name=graphName)
    # with open(fp) as f:
    edges = nx.read_edgelist(fp, comments="#", encoding="utf-8", nodetype=int)
    # nodes = nx.read_adjlist("nodes.txt")
    # my_graph.add_nodes_from(nodes)
    G.add_edges_from(edges.edges())

    return G


def draw_graph(G):

    # nx.draw(G, pos, with_labels=False, font_weight='bold')
    # labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
    plt.show()
    # plt.savefig("test.png")

def compute_adjacency(G):
    adj = nx.adjacency_matrix(G)
    return adj

def compute_laplacian(G):
    return nx.laplacian_matrix(G)

def compute_normalized_laplacian(G):
    return nx.normalized_laplacian_matrix(G)

def import_raw_data(graphName):
    pass


if __name__ =="__main__":
    print("Hello there, general Kenobi")
    G = create_graph("ca-AstroPh-reduced")

    # draw_graph(G)
    adj = compute_adjacency(G).todense()

    L = compute_laplacian(G)







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

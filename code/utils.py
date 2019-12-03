import networkx as nx
import os
import matplotlib.pyplot as plt

VERBOSE = True
DEBUG = True

def import_graph(graphName):
    """
    Import the graph from the txt file.
    """
    # Load txt file
    iprint("Importing graph data ...")
    fp = os.path.join("..", "graphs_processed", graphName + ".txt")

    # Create the graph
    G = nx.Graph(name=graphName)
    edges = nx.read_edgelist(fp, comments="#", encoding="utf-8", nodetype=int)
    G.add_edges_from(edges.edges())

    # Get info about the graph by parsing first line of txt file
    with open(fp) as f:
        firstLine = f.readline()
        fLineSplit = firstLine.split(" ")
        nVertices = int(fLineSplit[2])
        nEdges = int(fLineSplit[3])
        k = int(fLineSplit[4])
    nx.write_gpickle(G, "test.gpickle")
    return G, nVertices, nEdges, k


def draw_graph(G):
    """
    Draw the graph
    """
    # nx.draw(G, pos, with_labels=False, font_weight='bold')
    # labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
    plt.show()
    # plt.savefig("test.png")


def iprint(sth):
    """
    Info print
    :param sth:
    :return:
    """
    if VERBOSE is True:
        print(sth)

def dprint(sth):
    if DEBUG is True:
        print(sth)


# TODO: implement plots here

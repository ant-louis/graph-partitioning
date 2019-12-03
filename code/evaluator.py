import numpy as np
import sys
from utils import iprint, dprint
import matplotlib.pyplot as plt
import os
class Evaluator:
    """
    A class to evaluate some metrics of an algorithm or of a set of
    algorithm by performing a grid search to find the best suitable
    algorthm/parameters *for a given graph problem instance*.
    """

    def __init__(self, solver):
        self.solver = solver
        self.nVertices = solver.nVertices
        self.graphName = solver.G.name
        self.edges = solver.G.edges()
        self.k = solver.k

    def evaluate(self, output):
        """
        Compute some metrics for a given community partitioning and graph data
        :param output:
        :return:
        """
        k = output.shape[1]
        clusters = output[1, :] # partition
        # nodeIDs = output[0, :]
        nVerticesClusters = self._get_nVertices_per_cluster(clusters)
        minCSize = self._get_min_cluster_size(nVerticesClusters)
        maxCSize = self._get_max_cluster_size(nVerticesClusters)

        fCnt = self._get_frontiers(clusters)

        bindex = self._get_balance_index(nVerticesClusters, k)

        conductance = self._get_conductance(nVerticesClusters, fCnt)

        return {"name":self.graphName, "min_C_size":minCSize, "max_C_size":maxCSize,
                "conductance":conductance, "bindex":bindex}

    def _get_nVertices_per_cluster(self, clusters, stackNodeIDs=False):
        if stackNodeIDs is True:
            clusterIDs, cnts = np.unique(clusters, return_counts=True)
            return np.asarray((clusterIDs, cnts))
        else:
            _, cnts = np.unique(clusters, return_counts=True)
            return cnts

    def _get_frontiers(self, clusters):
        frontiers = np.zeros(self.k)
        for edge in self.edges:
            v1, v2 = edge
            if clusters[v1] != clusters[v2]:
                frontiers[clusters[v1]] += 1
                frontiers[clusters[v2]] += 1
        return frontiers

    def _get_min_cluster_size(self, nVerticesClusters):
        """
        Fast private implementation
        :param nVerticesClusters:
        :return: size of smallest community
        """
        return min(nVerticesClusters)


    def _get_max_cluster_size(self, nVerticesClusters):
        """
        Fast private implementation
        :param nVerticesClusters:
        :return: size of largest community
        """
        return max(nVerticesClusters)


    def get_min_cluster_size(self, clusters):
        """
        :return: size of smallest community
        """
        return min(self._get_nVertices_per_cluster(clusters))

    def get_max_cluster_size(self, clusters):
        """
        :return: size of largest community
        """
        return max(self._get_nVertices_per_cluster(clusters))


    def _get_balance_index(self, nVerticesClusters, k):
        return max(nVerticesClusters/(self.nVertices/k))

    def _get_conductance(self, nVerticesClusters, fCnt):
        """
        :param nVerticesClusters: number of vertices within each cluster
        :param fCnt: frontier count
        :return:
        """
        minCnts = np.array([min(cnt, self.nVertices - cnt) for cnt in
                               nVerticesClusters])
        return np.sum(fCnt / minCnts)


    def gridSearch(self, gridParams, dumpOutputBest=True, makeBarPlot=False):
        """
        Perform parameters optimization to find best algorithm for a given
        graph partitioning problem instance.
        :param dumpOutputBest: if we should write in a .txt the result of
        the best parameter set or not.
        :param makeBarPlot: if we should make a bar plot of the metrics results
        :return: (best parameters, bestMetrics, bestOutput), save output on
        fs if option is set to True
        """

        allMetrics = []
        bestOutput = None
        bestMetrics = {"conductance":float("inf")}
        bestParams = None
        iprint("\nPerforming grid search ...\n=========================")
        try:
            for i, params in enumerate(gridParams):

                iprint("\nAlgorithm {} with params = {}:\n-------------------------------------------------------\n".format(i, params))
                output = self.solver.make_clusters(params)
                metrics = self.evaluate(output)

                if metrics["conductance"] < bestMetrics["conductance"]:
                    bestOutput = output
                    bestMetrics = metrics
                    bestParams = params

                allMetrics.append(metrics)
        except Exception as e:
            sys.exit("Grid search failed: {}".format(e))


        print("\nEnd of gridsearch: best parameters were {} with "
              "metrics = {}".format(bestParams, bestMetrics))

        if dumpOutputBest is True:
            self.solver.dumpOutput(self.graphName, bestOutput)


        if makeBarPlot is True:
            self.barPlot(gridParams, allMetrics)

        return bestParams, bestMetrics, bestOutput

    def barPlot(self, allParams, allMetrics):

        labels = [str(p) for p in allParams]
        conductances = [m["conductance"] for m in allMetrics]
        index = np.arange(len(labels))

        fig, ax = plt.subplots()
        rects = ax.bar(index, conductances)

        # Add text on top of bar:
        self._autolabel(rects, ax)

        ax.set_xlabel('Set of parameters', fontsize=20)
        ax.set_ylabel('Conductance', fontsize=20)
        # ax.set_xticks(index, labels)
        # ax.xaxis.set_tick_params(labelsize=8, rotation=30)
        # ax.set_xticklabels(labels, fontsize=8) # , labelrotation=30
        plt.xticks(index, labels, fontsize=12) # , rotation=45

        plt.title('Conductance on {} per parameter sets'.format(
            self.graphName), fontsize=15)

        dirPath = os.path.dirname(os.path.realpath(__file__))
        dp = os.path.join(dirPath, "..", "plots")
        print(dp)
        fp = os.path.join(dp ,"barplot-"+self.graphName+".png")
        if not os.path.exists(dp):
            os.mkdir(dp)
        plt.savefig(fp)
        plt.show()

    def _autolabel(self, rects, ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
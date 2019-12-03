import numpy as np
import sys
from utils import iprint, dprint
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os

class Evaluator:
    """
    A class to evaluate some metrics of an algorithm or of a set of
    algorithm by performing a grid search to find the best suitable
    algorithm/parameters *for a given graph problem instance*.
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
        :param output: output of the solver.make_clustering method
        :return metrics: dictionary-like metrics
        """

        clusters = output[1, :] # partition
        # nodeIDs = output[0, :]
        nVerticesClusters = self._get_nVertices_per_cluster(clusters)
        minCSize = self._get_min_cluster_size(nVerticesClusters)
        maxCSize = self._get_max_cluster_size(nVerticesClusters)

        # Compute cut
        fCnts = self._get_frontiers(clusters)

        # Compute balance index
        bindex = self._get_balance_index(nVerticesClusters)

        # Compute normalized ratio cut
        nRatioCut = self._get_normalized_ratio_cut(clusters, fCnts)

        # Compute ratio cut
        score = self._get_ratio_cut(fCnts, nVerticesClusters)

        expansion = self._get_expansion(fCnts, nVerticesClusters)

        conductance = self._get_conductance(nVerticesClusters, fCnts)

        return {"name":self.graphName, "min_C_size":minCSize, "max_C_size":maxCSize,
                "n_ratio_cut":nRatioCut, "bindex":bindex, "score":score,
                "expansion":expansion, "conductance":conductance}

    def _get_nVertices_per_cluster(self, clusters, stackNodeIDs=False):
        if stackNodeIDs is True:
            clusterIDs, cnts = np.unique(clusters, return_counts=True)
            return np.asarray((clusterIDs, cnts))
        else:
            _, cnts = np.unique(clusters, return_counts=True)
            return cnts

    def _get_ratio_cut(self, fCnts, nVerticesClusters):
        """
        The objective function that we need to minimize in the statement.
        Minimum Ratio Cut: Given = (v, E), partition v into disjoint U and W
        such that e( U, W) /( | U | * | W|) is minimized with e(U, W) being
        the number of edges in {(u, w) in E | u in U and w in W}.
        Src: https://pdfs.semanticscholar.org/3627/8bf6919c6dced7d16dc0c02d725e1ed178f8.pdf
        """
        return np.sum(fCnts/nVerticesClusters)

    def _get_frontiers(self, clusters):
        """
        The "cut": numerator of the ratio cut
        """
        frontiers = np.zeros(self.k)
        for edge in self.edges:
            v1, v2 = edge
            if clusters[v1] != clusters[v2]:
                frontiers[clusters[v1]] += 1
                frontiers[clusters[v2]] += 1
        return frontiers

    def _get_min_cluster_size(self, nVerticesClusters):
        """
        :return: size of smallest community
        """
        return min(nVerticesClusters)


    def _get_max_cluster_size(self, nVerticesClusters):
        """
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

    def _get_balance_index(self, nVerticesClusters):
        """
        :return: balance index
        """
        return max(nVerticesClusters)/(self.nVertices/self.k)

    def _get_expansion(self, fCnts, nVerticesClusters):
        """
        src: https://fr.wikipedia.org/wiki/Taux_d%27expansion_(
        th%C3%A9orie_des_graphes)
        """

        return max(fCnts / nVerticesClusters)

    def _get_conductance(self, nVerticesClusters, fCnts):
        """
        src: https://en.wikipedia.org/wiki/Conductance_(graph)
        """
        denum = np.array(
            [min(cnt, self.nVertices - cnt) for cnt in nVerticesClusters])
        return np.sum(fCnts / denum)

    def _get_normalized_ratio_cut(self, clusters, fCnts):
        """
        Ratio between the number of cut edges and the volume of the smallest part
        """
        # minCnts = np.array([min(cnt, self.nVertices - cnt) for cnt in
        #                        nVerticesClusters])
        # return np.sum(fCnt / minCnts)
        degreeNodes = np.array(self.solver.adj.sum(axis=0))[0]
        degreeClusters = np.zeros(self.k)
        for nodeID, clusterID in enumerate(clusters):
            degreeClusters[clusterID] += degreeNodes[nodeID]
        cut = fCnts
        volume = degreeClusters
        # volume = np.multiply(degreeClusters, nVerticesClusters)
        return np.sum(cut/volume)




    def gridSearch(self, gridParams, dumpOutputBest=True, barPlots=("score")):
        """
        Perform parameters optimization to find best algorithm for a given
        graph partitioning problem instance.
        :param dumpOutputBest: if we should write in a .txt the result of
        the best parameter set or not.
        :param barPlot: a list of type of bar plots (metrics) that should be
        made to compare the algorithms
        :return: (best parameters, bestMetrics, bestOutput), save output on
        fs if option is set to True
        """

        allMetrics = []
        bestOutput = None
        bestMetrics = {"n_ratio_cut":float("inf"), "score": float("inf")}
        bestParams = None
        iprint("\nPerforming grid search on {} "
               "...\n=======================================".format(self.graphName))

        for i, params in enumerate(gridParams):

            iprint("\nAlgorithm {} with params = {}:\n-------------------------------------------------------\n".format(i+1, params))
            output = self.solver.make_clusters(params)
            metrics = self.evaluate(output)

            if metrics["score"] < bestMetrics["score"]:
                bestOutput = output
                bestMetrics = metrics
                bestParams = params
            print(metrics)
            allMetrics.append(metrics)


        print("\nEnd of gridsearch: best parameters were {} with "
              "metrics = {}".format(bestParams, bestMetrics))

        if dumpOutputBest is True:
            self.solver.dumpOutput(self.graphName, bestOutput)

        if barPlots is None or len(barPlots) == 0:
            return bestParams, bestMetrics, bestOutput
        else:
            if "score" in barPlots:
                y = [m["score"] for m in allMetrics]
                self.barPlot(y, gridParams, "Score")
            if "n_ratio_cut" in barPlots:
                y = [m["n_ratio_cut"] for m in allMetrics]
                self.barPlot(y, gridParams, "Normalized Ratio Cut")
            if "expansion" in barPlots:
                y = [m["expansion"] for m in allMetrics]
                self.barPlot(y, gridParams, "Expansion")
            if "bindex" in barPlots:
                y = [m["bindex"] for m in allMetrics]
                self.barPlot(y, gridParams, "Balance index")
            if "max_C_size" in barPlots:
                y = [m["max_C_size"] for m in allMetrics]
                self.barPlot(y, gridParams, "Maximum cluster size")
            if "min_C_size" in barPlots:
                y = [m["min_C_size"] for m in allMetrics]
                self.barPlot(y, gridParams, "Minimum cluster size")

            return bestParams, bestMetrics, bestOutput

    def barPlot(self, y, allParams, ylabel):

        labels = ["alg {}".format(i+1) for i in range(len(allParams))]

        index = np.arange(len(labels))
        fig, ax = plt.subplots()
        rects = ax.bar(index, y)

        plt.subplots_adjust(bottom=0.23)

        # Add text on top of bar:
        self._autolabel(rects, ax)

        ax.set_xlabel('Set of parameters', fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)

        idxTick = [r.get_x() + r.get_width() / 2 for r in rects]
        plt.xticks(idxTick, labels, fontsize=12, ha='center') # , rotation=45

        plt.title(ylabel+' on {} per parameter sets'.format(
            self.graphName), fontsize=15)

        dirPath = os.path.dirname(os.path.realpath(__file__))
        dp = os.path.join(dirPath, "..", "plots")

        fp = os.path.join(dp ,"barplot-"+ylabel.lower()+"-"+self.graphName+".png")
        if not os.path.exists(dp):
            os.mkdir(dp)
        plt.savefig(fp)
        plt.show()

    def _autolabel(self, rects, ax):
        """
        Attach a text label above each bar in *rects*, displaying its height.
        """
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{0:.3f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
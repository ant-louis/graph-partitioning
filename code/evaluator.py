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

        varCSize = np.var(nVerticesClusters)

        minCSize = self._get_min_cluster_size(nVerticesClusters)
        maxCSize = self._get_max_cluster_size(nVerticesClusters)

        # Compute cut
        cuts = self._get_cuts(clusters)

        # Compute balance index
        bindex = self._get_balance_index(nVerticesClusters)

        # Compute normalized ratio cut
        nRatioCut = self._get_normalized_ratio_cut(clusters, cuts)

        # Compute ratio cut
        score = self._get_ratio_cut(cuts, nVerticesClusters)

        expansion = self._get_expansion(cuts, nVerticesClusters)

        conductance = self._get_conductance(nVerticesClusters, cuts)

        return {"name":self.graphName,
                "score": score,
                "n_ratio_cut":nRatioCut,
                "bindex":bindex,
                "expansion":expansion,
                "conductance":conductance,
                "var_C_size":varCSize,
                "min_C_size":minCSize,
                "max_C_size":maxCSize}, nVerticesClusters

    def _get_nVertices_per_cluster(self, clusters, stackNodeIDs=False):
        if stackNodeIDs is True:
            clusterIDs, cnts = np.unique(clusters, return_counts=True)
            return np.asarray((clusterIDs, cnts))
        else:
            _, cnts = np.unique(clusters, return_counts=True)
            return cnts

    def _get_ratio_cut(self, cuts, nVerticesClusters):
        """
        The objective function that we need to minimize in the statement.
        Minimum Ratio Cut: Given = (v, E), partition v into disjoint U and W
        such that e( U, W) /( | U | * | W|) is minimized with e(U, W) being
        the number of edges in {(u, w) in E | u in U and w in W}.
        Src: https://pdfs.semanticscholar.org/3627/8bf6919c6dced7d16dc0c02d725e1ed178f8.pdf
        """
        return np.sum(cuts / nVerticesClusters)

    def _get_cuts(self, clusters):
        """
        The "cut": numerator of the ratio cut
        """
        cuts = np.zeros(self.k)
        for edge in self.edges:
            v1, v2 = edge
            if clusters[v1] != clusters[v2]:
                cuts[clusters[v1]] += 1
                cuts[clusters[v2]] += 1
        return cuts

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

    def _get_expansion(self, cuts, nVerticesClusters):
        """
        src: https://fr.wikipedia.org/wiki/Taux_d%27expansion_(
        th%C3%A9orie_des_graphes)
        """

        return max(cuts / nVerticesClusters)

    def _get_conductance(self, nVerticesClusters, cuts):
        """
        src: https://en.wikipedia.org/wiki/Conductance_(graph)
        """
        denum = np.array(
            [min(cnt, self.nVertices - cnt) for cnt in nVerticesClusters])
        return np.sum(cuts / denum)

    def _get_normalized_ratio_cut(self, clusters, cuts):
        """
        Ratio between the number of cut edges and the volume of the smallest part
        src: [Shi and Malik, 2000]
        """
        degreeNodes = np.array(self.solver.adj.sum(axis=0))[0]
        degreeClusters = np.zeros(self.k)
        for nodeID, clusterID in enumerate(clusters):
            degreeClusters[clusterID] += degreeNodes[nodeID]

        volume = degreeClusters
        # volume = np.multiply(degreeClusters, nVerticesClusters)
        return np.sum(cuts/volume)




    def gridSearch(self, gridParams, dumpOutputBest=True, plots=("score")):
        """
        Perform parameters optimization to find best algorithm for a given
        graph partitioning problem instance.
        :param dumpOutputBest: if we should write in a .txt the result of
        the best parameter set or not.
        :param plots: a list of metrics to plot to compare the algorithms
        :return: (best parameters, bestMetrics, bestOutput), save output on
        fs if option is set to True
        """

        allMetrics = []
        bestOutput = None
        bestMetrics = {"n_ratio_cut":float("inf"), "score": float("inf")}
        bestParams = None
        allClusterSizes = []
        iprint("\nPerforming grid search on {} "
               "...\n=======================================".format(self.graphName))

        for i, params in enumerate(gridParams):

            iprint("\nAlgorithm {} with params = {}:\n-------------------------------------------------------\n".format(i+1, params))
            output = self.solver.make_clusters(params)
            metrics, nVerticesClusters = self.evaluate(output)

            if metrics["score"] < bestMetrics["score"]:
                bestOutput = output
                bestMetrics = metrics
                bestParams = params
            print(metrics)
            allMetrics.append(metrics)
            allClusterSizes.append(nVerticesClusters)


        print("\nEnd of gridsearch: best parameters were {} with "
              "metrics = {}".format(bestParams, bestMetrics))

        if dumpOutputBest is True:
            self.solver.dumpOutput(self.graphName, bestOutput)

        if plots is None or len(plots) == 0:
            return bestParams, bestMetrics, bestOutput
        else:
            if "score" in plots:
                y = [m["score"] for m in allMetrics]
                self.barPlot(y, gridParams, "Score")
            if "n_ratio_cut" in plots:
                y = [m["n_ratio_cut"] for m in allMetrics]
                self.barPlot(y, gridParams, "Normalized-Ratio-Cut")
            if "expansion" in plots:
                y = [m["expansion"] for m in allMetrics]
                self.barPlot(y, gridParams, "Expansion")
            if "bindex" in plots:
                y = [m["bindex"] for m in allMetrics]
                self.barPlot(y, gridParams, "Balance index")
            if "max_C_size" in plots:
                y = [m["max_C_size"] for m in allMetrics]
                self.barPlot(y, gridParams, "Maximum cluster size")
            if "min_C_size" in plots:
                y = [m["min_C_size"] for m in allMetrics]
                self.barPlot(y, gridParams, "Minimum cluster size")
            if "var_C_size" in plots:
                y = [m["var_C_size"] for m in allMetrics]
                self.barPlot(y, gridParams, "Variance of cluster size")

            if "box_plot" in plots:
                self.boxPlot(allClusterSizes)

            return bestParams, bestMetrics, bestOutput

    def boxPlot(self, allClusterSizes):
        allClusterSizes = np.array(allClusterSizes).T
        red_square = dict(markerfacecolor='r', marker='s')
        green_square = dict(markerfacecolor='g', marker='D')
        fig, ax = plt.subplots()
        bp = ax.boxplot(allClusterSizes, showmeans= True,
                        flierprops=red_square, meanprops=green_square, showfliers=False)
        labels = ["Algo {}".format(i+1) for i in range(
            allClusterSizes.shape[1])]

        ax.set_xticklabels(labels)
        ax.set_ylabel("Cluster size", fontsize=20)
        # ax.set_xlabel('Set of parameters', fontsize=20)
        # ax.set_title('Box plot of cluster sizes', fontsize=15)

        dirPath = os.path.dirname(os.path.realpath(__file__))
        dp = os.path.join(dirPath, "..", "plots")

        fp = os.path.join(dp, "boxplot-"+self.graphName+ ".png")
        if not os.path.exists(dp):
            os.mkdir(dp)
        plt.savefig(fp)
        plt.show()


    def barPlot(self, y, allParams, ylabel):

        labels = ["alg {}".format(i+1) for i in range(len(allParams))]

        index = np.arange(len(labels))
        fig, ax = plt.subplots()
        rects = ax.bar(index, y)

        plt.subplots_adjust(bottom=0.23)

        # Add text on top of bar:
        self._autolabel(rects, ax)

        # ax.set_xlabel('Set of parameters', fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)

        idxTick = [r.get_x() + r.get_width() / 2 for r in rects]
        plt.xticks(idxTick, labels, fontsize=12, ha='center') # , rotation=45

        # plt.title(ylabel+' on {} per parameter sets'.format(
        #     self.graphName), fontsize=15)

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
            ax.annotate('{0:.4f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
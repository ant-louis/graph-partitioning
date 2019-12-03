import numpy as np


class Evaluator:
    """
    A class to evaluate some metrics of an algorithm or of a set of
    algorithm by performing a grid search to find the best suitable
    algorthm/parameters *for a given graph problem instance*.
    """

    def __init__(self, solver):
        self.solver = solver
        self.nVertices = solver.nVertices
        self.name = solver.G.name
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

        return {"name":self.name, "min_C_size":minCSize, "max_C_size":maxCSize,
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
        Perform parameter optimization to find best algorithm with the right set of parameters
        :param dumpOutputBest: if we should write in a .txt only the result
        of the best parameter set or not.
        :return: best parameters dictionary, save files on fs
        """

        # TODO: iterate over gridParam
        output = self.solver.partition()
        print("Best parameters were {} with algo {}")
        self.solver.dumpOutput("{}-{}-{}".format(self.name, 1, 2),
                               output)


        if makeBarPlot is True:
            # TODO make bar plot of score of each parameter set
            pass

        bestParam = {"sth":3, "sth2":3}
        return bestParam

    def plot(self, sth):
        # TODO
        pass


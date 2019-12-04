# Sources

* [Spectral Modification of Graphs for Improved Spectral Clustering](https://papers.nips.cc/paper/8732-spectral-modification-of-graphs-for-improved-spectral-clustering)
    * Spectral clustering algorithms provide approximate solutions to hard optimization problems that formulate graph partitioning in terms of the graph conductance. It is well understood that the quality of these approximate solutions is negatively affected by a possibly significant gap between the conductance and the second eigenvalue of the graph. In this paper we show that for \textbf{any} graph  , there exists a `spectral maximizer' graph   which is cut-similar to  , but has eigenvalues that are near the theoretical limit implied by the cut structure of  . Applying then spectral clustering on   has the potential to produce improved cuts that also exist in   due to the cut similarity. This leads to the second contribution of this work: we describe a practical spectral modification algorithm that raises the eigenvalues of the input graph, while preserving its cuts. Combined with spectral clustering on the modified graph, this yields demonstrably improved cuts.
    * Roughly  speaking,  spectral  clusteringcomputes the second eigenvalueλof the normalized graph Laplacian as an approximation to thegraph conductance, i.e. the value of the optimal cut
    
*
*
* 
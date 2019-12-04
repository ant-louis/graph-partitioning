# Sources
* [Le 3e algo de spectral clustering](https://dl.acm.org/citation.cfm?id=946658)
* [Spectral Modification of Graphs for Improved Spectral Clustering](https://papers.nips.cc/paper/8732-spectral-modification-of-graphs-for-improved-spectral-clustering)
    * Spectral clustering algorithms provide approximate solutions to hard optimization problems that formulate graph partitioning in terms of the graph conductance. It is well understood that the quality of these approximate solutions is negatively affected by a possibly significant gap between the conductance and the second eigenvalue of the graph. In this paper we show that for \textbf{any} graph  , there exists a `spectral maximizer' graph   which is cut-similar to  , but has eigenvalues that are near the theoretical limit implied by the cut structure of  . Applying then spectral clustering on   has the potential to produce improved cuts that also exist in   due to the cut similarity. This leads to the second contribution of this work: we describe a practical spectral modification algorithm that raises the eigenvalues of the input graph, while preserving its cuts. Combined with spectral clustering on the modified graph, this yields demonstrably improved cuts.
    * Roughly  speaking,  spectral  clusteringcomputes the second eigenvalueλof the normalized graph Laplacian as an approximation to thegraph conductance, i.e. the value of the optimal cut
    *
    
    *  Itis well understood that the quality of these approximate solutions is negativelyaffected by a possibly significant gap between the conductance and the secondeigenvalue of the graph. In this paper we show that for any graphG, there existsa ‘spectral maximizer’ graphHwhich is cut-similar toG, but has eigenvaluesthat are near the theoretical limit implied by the cut structure ofG. Applying thenspectral clustering onHhas the potential to produce improved cuts that also existinGdue to the cut similarity. This leads to the second contribution of this work:we describe a practical spectral modification algorithm that raises the eigenvaluesof the input graph, while preserving its cuts. Combined with spectral clustering onthe modified graph, this yields demonstrably improved cuts
    * Utilisation d'un graph maximizer M qui approxime le graph H qui 
    optimize au mieux le spectral clustering
* [Improved Spectral Clustering Algorithm Based on Similarity Measure](https://link.springer.com/chapter/10.1007/978-3-319-14717-8_50)
    * this paper proposed to utilize the similarity measure based on data density during creating the similarity matrix, inspired by density sensitive similarity measure. Making it increase the distance of the pairs of data in the high density areas, which are located in different spaces. And it can reduce the similarity degree among the pairs of data in the same density region, so as to find the spatial distribution characteristics complex data.
    * In addition to matching spectral clustering algorithm, the final stage of the algorithm is to use the k-means (or other traditional clustering algorithms) for the selected feature vector to cluster, however the k-means algorithm is sensitive to the initial cluster centers. Therefore, we also designed a simple and effective method to optimize the initial cluster centers leads to improve the k-means algorithm, and applied the improved method to the proposed spectral clustering algorithm
    
* [Improved spectral clustering using PCA based similarity measure on different Laplacian graphs]()
    * n this paper we implement PCA based similarity measure for graph construction
    * n PCA based similarity measure, the similarity measure based on eigenvalues and its eigenvectors is used for building the graph and we study the efficiency of two types of Laplacian graph matrices. This graph is then clustered using spectral clustering algorithm
* [A Probabilistic Approach for OptimizingSpectral Clustering](https://pdfs.semanticscholar.org/a8b4/d0cf79be49aa9c1d482876e5c34ac1b7e063.pdf)
    * In this paper, we present a new spectral clustering algorithm, named “Soft Cut”, that ex-plicitly  addresses  the  above  two  problems.   It  extends  the  normalized  cut  algorithm  byintroducing probabilistic membership of data points.  By encoding membership of multi-ple clusters into a set of probabilities,  the proposed clustering algorithm can be applieddirectly to multi-class clustering problems. Our empirical studies with a variety of datasetshave shown that the soft cut algorithm can substantially outperform the normalized cutalgorithm for multi-class clustering
    * New algo= soft-cut
    
* [https://hal.archives-ouvertes.fr/hal-01516649/file/Paper_PowerSpectral
.pdf](https://hal.archives-ouvertes.fr/hal-01516649/file/Paper_PowerSpectral.pdf)
    * Power spectral clustering
    * in this article, the aim is to develop an algorithm which uses  
    efficient  MST  based  clustering  within  a  cluster  andmore  computationally  expensive  spectral  clustering  nearthe  borders  of  the  clusters.

* [A Spectral Clustering Approach To Finding Communities in Graph](http://www.datalab.uci.edu/papers/siam_graph_clustering.pdf)
    * Pas besoin de choisir k dès le départ
 
 
 ## min-cut algo
 
 By contrast, the min-cut clustering is realized by constructinga weighted undirected graph and then partitioning its vertices intotwo sets so that the total weight of the set of edges with endpointsin different sets is minimized [16] [17].   
    
A straightforward solution of this problem is to project theoriginal 
dataset to a low-dimensional subspace by dimensionalityreduction, for 
example, PCA, before performing Traditional k-mean

Among several graph clus-tering methods, min-cut tends to provide more balanced clusters ascompared to other graph clustering criterion. As the within-clustersimilarity in min-cut method is explicitly maximized, solving themin-cut clustering problem is nontrivial

## Seems less interesting:

* [Improving Spectral Clustering using the Asymptotic Value of the 
Normalised Cut](https://arxiv.org/abs/1703.09975)
* [Scalable Clustering of Signed Networks Using BalanceNormalized Cut](https://www.cs.utexas.edu/~inderjit/public_papers/sign_clustering_cikm12.pdf)
* [An improved spectral clustering algorithm based on random walk](https://www.researchgate.net/publication/220412587_An_improved_spectral_clustering_algorithm_based_on_random_walk)
* [Spectral Clustering based on the graphp-Laplacian](https://www.icml.cc/Conferences/2009/papers/377.pdf)
    * Bon résumé dse métriques mais c'est tout



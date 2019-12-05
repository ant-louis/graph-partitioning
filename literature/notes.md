## Idées

* Pre-processing du graphe (densité des noeuds, ...)
* Autre types de laplacien ou spectral analysis
* Autre algo que kmean
* Probabilitic clustering au lieu de hard-clustering
    * Less likely to be trapped in local miniames
    * ex: bayesian clustering method
* Utiliser MST (Maximum  Spanning  Tree) pour milieu du cluster et spectral 
pour bordure du cluster -> [power spectral clustering](https://hal.archives-ouvertes.fr/hal-01516649/file/Paper_PowerSpectral.pdf)






## Notes

According to [this paper](https://citeseerx.ist.psu
.edu/viewdoc/summary?doi=10.1.1.19.8100),  row-normalizing the matrixof eigenvectors, so that the row vectors are projectedonto the unit hypersphere, gives much higher qualityresults.


Therefore, ratio cut is the special case of normalized cutwhereD=I

It is well known that the second eigenvectorsof the unnormalized and 
normalized graph Laplacianscorrespond  to  relaxations  of  the  ratio  cut 
and normalized cut

The usual objective for sucha partitioning is to have high within-cluster similarityand low inter-cluster similarity



One common approach is to first construct a low-dimension space for data repre-sentation using the smallest eigenvectors of a graph Laplacian that is constructed based onthe pair wise similarity of data. Then, a standard clustering algorithm, such as the K-meansmethod, is applied to cluster data points in the low-dimension space


 A too small number of eigenvectors will lead to an insufficient representation ofdata, and meanwhile a too large number of eigenvectors will bring in a significant amountof noise to the data representation.  Both cases will degradethe quality of clustering.  Al-though it has been shown in (Ng et al., 2001) that the number ofrequired eigenvectors isgenerally equal to the number of clusters, the analysis is valid only when data points ofdifferent clusters are well separated. As will be shown later, when data points are not wellseparated, the optimal number of eigenvectors can be different from the number of clusters
 
 
 
 Another problem with the existing spectral clustering algorithms is that they are based onbinary cluster membership and therefore are unable to express the uncertainty in data clus-tering.  Compared to hard cluster membership, probabilistic membership is advantageousin that it is less likely to be trapped by local minimums.
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
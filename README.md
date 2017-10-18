# Tool for clustering embeddings

So far, just a simple script to run a bunch of sklearn clustering algorithms
on embeddings in word2vec format.

Some results on usefulness:
* usefulness for POS tagging: https://github.com/GateNLP/exp-lf-pos

## Usage

Get usage information by running:
* `python3 ./python/cluster-embs.py -h`

Currently the following clustering algorithms are supported (using the sklearn back-end). For 
some of the algorithms, information about the clusters is written to a file with extension `info.json`:
* MiniBatchKMeans (default): k-means clustering using minibatch SGD
  Info: `cluster_centers`, `inertia`, `counts`, `n_iter`
* KMeans: k-means clustering
  Info: `cluter_centers`, `inertia`, `n_iter`
* AgglomerativeClusteringWard: agglomerative clustering using ward linkage
* AgglomerativeClusteringAverageEuclidean: agglomerative clustering using average linkage 
* Birch
* SpectralClustering: spectral clustering with rbf affinity 

For MiniBatchKMeans and KMeans, the 20 elements most similar to each of the k cluster centroids 
are stored in a file with the extension `mostsimilar.json`.

## Clustering times

| embeddings | machine | alg | k | elapsed time, total | elapsed time, clustering | 
|------------|---------|-----|---|-------------|----|
| fasttext wiki.en.vec | derwent | KMeans | 100 | 5:21:16 | ??? |
| fasttext wiki.en.vec | derwent | MiniBatchKMeans | 100 |  0:16:37 | 0:00:38 |
| fasttext wiki.en.vec | derwent | MiniBatchKMeans | 500 |  0:22:18 | 0:02:22 |
| fasttext wiki.bg.vec | derwent | MiniBatchKMeans | 100 |  0:02:27 | 0:00:13 | 
| fasttext wiki.bg.vec | derwent | MiniBatchKMeans | 500 |  0:04:42 | 0:01:18 |  


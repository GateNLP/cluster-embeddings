# Tool for clustering embeddings

So far, just a simple script to run a bunch of sklearn clustering algorithms
on embeddings in word2vec format.

Some results on usefulness:
* usefulness for POS tagging: https://github.com/GateNLP/exp-lf-pos


## Clustering times

| embeddings | machine | alg | k | elapsed time, total | elapsed time, clustering | 
|------------|---------|-----|---|-------------|----|
| fasttext wiki.en.vec | derwent | KMeans | 100 | 5:21:16 | ??? |
| fasttext wiki.en.vec | derwent | MiniBatchKMeans | 100 |  0:16:37 | 0:00:38 |
| fasttext wiki.en.vec | derwent | MiniBatchKMeans | 500 |  0:22:18 | 0:02:22 |
| fasttext wiki.bg.vec | derwent | MiniBatchKMeans | 100 |  0:02:27 | 0:00:13 | 
| fasttext wiki.bg.vec | derwent | MiniBatchKMeans | 500 |  0:04:42 | 0:01:18 |  


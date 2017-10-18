# Tool for clustering embeddings

So far, just a simple script to run a bunch of sklearn clustering algorithms
on embeddings in word2vec format.

## Clustering times

Of facebook embeddings (old runs):
* wiki.en.vec, k=100: 19276 seconds (on derwent), 44488 seconds (on zeus)
* wiki.en.vec, k=500:   (on derwent)

Of facebook embeddings (recent runs):
* wiki.en.vec, mbknn, k=100: 16:37 total,  38 secs for clustering (derwent)
* wiki.en.vec, mbknn, k=500: 22:18 total, 142 secs for clustering (derwent)
* wiki.bg.vec, mbknn, k=100:  2:27 total,  13 secs for clustering (derwent)
* wiki.bg.vec, mbknn, k=500:  4:42 total,  78 secs for clustering (derwent) 

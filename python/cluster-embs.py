
## Simple script for clustering embeddings

## NOTE: this probably only works with python 3.5 or later

import sys
import gensim
import sklearn.cluster
import argparse
import time
# import json

nan = float('nan')

parser = argparse.ArgumentParser(description="Cluster embeddings")
parser.add_argument("-v", action='store_true', help="Show more messages about what the program is doing.")
parser.add_argument("-d", action='store_true', help="Show debug messages")
parser.add_argument("inFile", nargs=1, help="The embedding file to read.")
## parser.add_argument("outFilePrefix", nargs=1, help="The prefix for all the files that will be written")
parser.add_argument("-m", nargs=1, type=str, help="The clustering algorithm to use (default is MiniBatchKMeans)")
parser.add_argument("-k", nargs=1, type=int, help="Number of clusters if the method needs this parameter (default is 100)")
parser.add_argument("-s", nargs=1, type=int, help="Random seed (default is 1)")

args = parser.parse_args()

verbose = args.v

debug = args.d
if debug:
    verbose = True

inFile = args.inFile[0]
#outFilePrefix = args.outFile[0]

method = "MiniBatchKMeans"
if args.m:
    method = args.m[0]

seed = 1
if args.s:
    seed = int(args[0])
k = 100
if(args.k):
    k=int(args.k[0])


inBinary = ".bin." in inFile
if verbose: print("Loading embeddings file ", inFile," using binary format: ",inBinary, file=sys.stderr)
embs = gensim.models.Word2Vec.load_word2vec_format(inFile, binary=inBinary, unicode_errors='ignore', encoding='utf8')
print("Embeddings file loaded,  shape (nWords, nDims): ",embs.syn0.shape)
if debug: print("DEBUG: embeddings for 'mother': ",embs["mother"],file=sys.stderr)

nEmbeddings = len(embs.index2word)

if method == "MiniBatchKMeans":
    clmethod = sklearn.cluster.MiniBatchKMeans(n_clusters=k,random_state=seed)
elif method == "KMeans":
    clmethod = sklearn.cluster.KMeans(n_clusters=k, random_state=seed)
elif method == "AgglomerativeClusteringWard":
    clmethod = sklearn.cluster.AgglomerativeClustering(n_clusters=k, linkage="ward")
elif method == "AgglomerativeClusteringAverageEuclidean":
    clmethod = sklearn.cluster.AgglomerativeClustering(n_clusters=k, linkage="average", affinity="euclidean")
elif method == "Birch":
    clmethod = sklearn.cluster.Birch(n_clusters=k)
else:
    exit("Not a valid clustering algorithm: "+method)
if verbose:
    print("Running clustering ",method,file=sys.stderr)
start = time.time()
clids = clmethod.fit_predict(embs.syn0)
end = time.time()
if verbose: print("Clustering completed, needed ","{:.2f} seconds".format(end-start),file=sys.stderr)

if verbose: print("Writing cluster ids to stdout",file=sys.stderr)
for i in range(len(clids)):
    print(embs.index2word[i], "\t", clids[i], sep='')

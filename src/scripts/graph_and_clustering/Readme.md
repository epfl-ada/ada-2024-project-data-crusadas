# Graph and Clustering

This folder contains the scripts used to generate the graph and clustering of the data.

Here are the steps we followed:

- We first tokenize all the reviews and create an inverted index (a dictionary where the keys are the words and the values are the list of reviews where the word appears).
- We first keep only the 1000 most frequent words.
- For each of these words and for each beer, we compute the number of times the word appears in the reviews of the beer.
- We further filter the words based on their coefficient of variation (CV) and keep only the words with a high CV. This is done to keep only the words that are discriminative, i.e., words that are used differently across the beers (e.g., "citrus") and not words that are used similarly across the beers (e.g., "beer").
- We then compute the cosine similarity between the beers based on the words that we kept.
- We use the cosine similarity to create a graph where the nodes are the beers and the edges are the cosine similarity between the beers. For computational reasons, we only keep edges with a cosine similarity above a certain threshold.
- We then use an algorithm to cluster the beers based on the graph. We use the Louvain algorithm, which is a modularity-based method for community detection in networks.
- We then visualize the graph and the clustering.

A lot of these steps are very computationally intensive. In particular the computation of the cosine similarity matrix, the creation of the graph and the clustering and the visualization of the graph. For these three steps, we optimized the code as much as possible in memory and in time. We parallelized the code as much as possible using mpi4py. We ran the scripts on an HPC cluster (see the corresponding batch files).

The scripts are organized as follows:
- `similarity_graph.ipynb`: Jupyter notebook that contains the code for non intensive computations (data processing, etc.) and prototyping all steps of the pipeline (computing cosine similarity, creating graph, clustering, etc.) with a small subset of the data.
- `compute_cosine_similarity.py`: Script that computes the cosine similarity between the beers based on the words that we kept.
- `compute_louvain_cluster.py`: Script that clusters the beers based on the graph.
- `visualize_graph.py`: Script that visualizes the graph and the clustering.
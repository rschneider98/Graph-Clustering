/* Richard Schneider
   Graph-Clustering.cpp
   Description: This project is used for educational purposes. It is designed 
   to consume a graph dataset from a file, convert it into a matrix, and then
   cluster the vertices based on the Spectral Clustering algorithm which
   uses the Eigenvector centrality metric for each of the Eigenvalues to
   convert the graph clustering problem into a vector-based problem. */

#include <iostream>
#include <utility> 
#include <vector>
#include "LinAlg.h"
#include "Graph.h"

int main()
{
	// read in the matrix as a graph
	Graph::Graph test1("../Data/power-494-bus.mtx", true);
	std::pair<LinAlg::Matrix, LinAlg::Vector> clusters = test1.EigenClustering(7);
	clusters.first.toFile("../Data/coords.mtx");
	clusters.second.toFile("../Data/clusters.txt");
	return 0;
}

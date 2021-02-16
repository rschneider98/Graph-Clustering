#pragma once

#include <fstream>
#include <string>
#include <sstream>
#include "LinAlg.h"

namespace Graph {

class Graph {
private:
	LinAlg::Matrix adj;
public:
	Graph(std::string filename, bool isSymmetric) {
		/* This function takes in a filename and reads in the matrix from 
		the (i, j, value) tuples. */
		// check if file exists
		std::ifstream infile;
		infile.open(filename, std::ios::in);
		if (infile.is_open()) {
			// read lines
			int max = 0;
			std::string line;
			while (std::getline(infile, line))
			{
				// skip comments in file (uses %)
				if (line[0] != '%') {
					// create string stream
					std::istringstream iss(line);
					int i, j;
					double value;
					if (!(iss >> i >> j >> value)) { break; } // error
					// find max vertex number
					if (i > max) { max = i; }
					if (j > max) { max = j; }
				}
			}
			// create square matrix
			adj = LinAlg::Matrix(max, max);
			// reread file and add values to matrix
			infile.seekg(0);
			while (std::getline(infile, line))
			{
				// skip comments in file (uses %)
				if (line[0] != '%') {
					// create string stream
					std::istringstream iss(line);
					int i, j;
					double value;
					if (!(iss >> i >> j >> value)) { break; } // error
					// find max vertex number
					adj[i][j] = value;
					if (isSymmetric) {
						adj[j][i] = value;
					}
				}
			}
			// close file
			infile.close();
		}
	}
	LinAlg::Vector EigenClustering(int k) {
		/* This function takes the adj matrix, finds the 
		eigenvectors, and clusters the graph into k parts*/
		LinAlg::Matrix eigenvects = adj.Eigenvectors();
		return eigenvects.KMeans(k);
	}
};

}
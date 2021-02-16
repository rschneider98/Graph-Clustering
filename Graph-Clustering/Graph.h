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
		bool isFirstLine = true;
		std::ifstream infile;
		infile.open(filename, std::ios::in);
		if (infile.is_open()) {
			// read lines
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
					// first line contains a different list, it describes the matrix 
					// size and number of elements - i, j, length
					if (isFirstLine) {
						adj = LinAlg::Matrix(i, j);
						isFirstLine = false;
					}
					else {
						// add entry to the matrix
						adj[i - 1][j - 1] = value;
						if (isSymmetric) {
							adj[j - 1][i - 1] = value;
						}
					}
				}
			}
			infile.close();
		}
	}

	friend std::ostream& operator<<(std::ostream& out, Graph& rhs) {
		/* Print out the matrix */
		out << rhs.adj;
		return out;
	}
	std::vector<double> operator[](int i) {
		return adj[i];
	}

	std::pair<LinAlg::Matrix, LinAlg::Vector> EigenClustering(int k) {
		/* This function takes the adj matrix, finds the 
		eigenvectors, and clusters the graph into k parts*/
		// find eigenvectors and cluster vectors
		std::pair<LinAlg::Matrix, LinAlg::Matrix> eig_pair = adj.QRAlgorithm();
		LinAlg::Matrix eigenvects = eig_pair.first;
		LinAlg::Matrix eigenval = eig_pair.second;
		LinAlg::Vector clusters = eigenvects.KMeans(k);

		// find two largest eigenvalues and make coordinate matrix for output
		double max1 = eigenval[0][0];
		double max2 = eigenval[1][1];
		int loc1 = 0;
		int loc2 = 1;
		if (max2 > max1) {
			std::swap(max1, max2);
			std::swap(loc1, loc2);
		}
		for (int i = 2; i < eigenval[0].size(); i++) {
			if (max1 < eigenval[i][i]) {
				loc1 = i;
				max1 = eigenval[i][i];
				loc2 = loc1;
				max2 = max1;
			}
			else if (max2 < eigenval[i][i]) {
				max2 = eigenval[i][i];
				loc2 = i;
			}
		}
		LinAlg::Matrix coord(std::vector<LinAlg::Vector>{eigenvects.GetCol(loc1), eigenvects.GetCol(loc2)});

		// get vector of real eigenvalues
		LinAlg::Vector eig(eigenval[0].size(), 0);
#pragma omp parallel for
		for (int i = 0; i < eigenval[0].size(); i++) {
			eig[i] = eigenval[i][i];
		}
		return std::make_pair(coord, clusters);
	}
};

}
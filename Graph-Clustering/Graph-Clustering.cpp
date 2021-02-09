/* Richard Schneider
   Graph-Clustering.cpp
   Description: This project is used for educational purposes. It is designed 
   to consume a graph dataset from a file, convert it into a matrix, and then
   cluster the vertices based on the Spectral Clustering algorithm which
   uses the Eigenvector centrality metric for each of the Eigenvalues to
   convert the graph clustering problem into a vector-based problem. */

#include <iostream>
#include <vector>
#include "LinAlg.h"

int main()
{
	std::vector<std::vector<double>> myVect1{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9}
	};

	std::vector<std::vector<double>> myVect2{
		{1, 2, 3},
		{0, 5, 6},
		{0, 0, 9}
	};

	std::vector<std::vector<double>> myVect3{
		{0.73054, 0.715638, 0.673198, 0.0587931},
		{0.189321, 0.447365, 0.596879, 0.448019},
		{0.908993, 0.847864, 0.237653, 0.708224}
	};

	std::vector<std::vector<double>> myVect4{
		{0.751884},
		{0.039623},
		{0.811679},
		{0.919456}
	};
	std::vector<std::vector<double>> myVect5{
		{0.83462, 0.699265, 0.930962, 0.863479},
		{0.473505, 0.691988, 0.747999, 0.225765},
		{0.933097, 0.105651, 0.280316, 0.0671968},
		{0.496422, 0.556214, 0.727784, 0.789878}
	};
	std::vector<std::vector<double>> myVect6{
		{1, -5, 8, 3},
		{1, -2, 1, 2},
		{2, -1, -5, 6}
	};
	std::vector<std::vector<double>> myVect7{
		{1, -5, 8},
		{1, -2, 1},
		{2, -1, -5},
		{2, 4, 7}
	};
	std::vector<std::vector<double>> myVect8{
		{1}, {2}, {3}
	};
	std::vector<std::vector<double>> myVect9{
		{1, 2, 3},
		{5, 7, 9},
		{11, 13, 17}
	};
	std::vector<double> myVect10{ 1, 2, 3 };
	std::vector<double> myVect12{ 1, 0, 0 };

	std::vector<std::vector<double>> myVect11{
		{0.749935, 0.0239392, 0.909929},
		{0.233785, 0.0521392, 0.192435},
		{0.632099, 0.611084, 0.748108}
	};

	std::vector<std::vector<double>> myVect13{
		{3, 2},
		{1, 2}
	};

	LinAlg::Matrix test1(myVect11);
	LinAlg::Matrix test2 = test1.GramSchmidt();
	std::cout << test1 << std::endl;
	std::cout << test2;
	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

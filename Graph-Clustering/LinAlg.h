#pragma once
/* Richard Schneider
   24 JAN 2021
   Matrix Structure and Algorithms for Spectral Graph Clustering 
   FOR EDUCATIONAL PURPOSES, this is not designed or tested for use
   beyond algorithm implementation. */

#include <iostream>
#include <exception>
#include <vector>
#include <omp.h>

namespace LinAlg {

// exceptions used in this namespace
class ZeroRows : public std::exception {
	virtual const char* what() const throw() {
		return "A Matrix declared with a vector must have at least one row";
	}
};

class ZeroCols : public std::exception {
	virtual const char* what() const throw() {
		return "A Matrix declared with a vector must have at least one column";
	}
};

class UnevenCols : public std::exception {
	virtual const char* what() const throw() {
		return "A Matrix declared with a vector must have columns of equal sizes";
	}
};

class UnequalDim : public std::exception {
	virtual const char* what() const throw() {
		return "Matrices must be of equal dimensions for addition";
	}
};

class IncorrectDim : public std::exception {
	virtual const char* what() const throw() {
		return "For matrix multiplication, the number of rows for left-hand side matrix must be equal to the number of columns of the right-hand matrix";
	}
};

class NotSquare : public std::exception {
	virtual const char* what() const throw() {
		return "Only square matries can have a determinant";
	}
};

class NoInverse : public std::exception {
	virtual const char* what() const throw() {
		return "Only square matries that are linearly independent can have a non-zero determinant and an inverse";
	}
};

// Matrix class since that is the base data structure of Linear Algebra
class Matrix {
protected:
	// vector of vectors of doubles to represent the matrix in row-major form
	std::vector<std::vector<double>> data;
	// dimension of the matrix
	int m, n;

public:
	// empty constructor
	Matrix() {}
	// constructor with shape (defaults to zeros)
	Matrix(int m, int n) {
		this->m = m;
		this->n = n;
		data = std::vector<std::vector<double>> (m, std::vector<double>(n, 0));
	}
	// constructor based on passing in the data
	Matrix(std::vector<std::vector<double>> data) {
		// get number of rows 
		m = data.size();
		// verify there is at least one row
		if (m == 0) {
			throw ZeroRows();
		}
		// get the number of columns for the first row
		n = data[0].size();
		// verify there is at least one column
		if (n == 0) {
			throw ZeroCols();
		}
		// verify each row has the same number of columns
		for (int i = 0; i < m; i++) {
			if (data[i].size() != n) {
				throw UnevenCols();
			}
		}
		// we have validated the input data and can keep it
		this->data = data;
	}
		

	// data access operator overloading
	std::vector<double>& operator[](int i) {
		/* This overloads the access operator to only use one bracket instead of two, and it
		automatically references the data vectors */
		return data[i];
	}
	double& at(int i, int j) {
		/* Returns tha data at location i, j */
		return data[i][j];
	}


	// arithmetic operator overloading
	Matrix operator+(Matrix &rhs) {
		/* This function takes in another matrix, sums this and the rhs element-wise, and returns a new matrix*/
		// verify both matrices are same dimenison
		if ((n != rhs.n) || (m != rhs.m)) {
			throw UnequalDim();
		}
		// create output matrix, initailized to zeros
		Matrix out(m, n);
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				out[i][j] = data[i][j] + rhs[i][j];
			}
		}

		return out;
	}
	Matrix operator-(Matrix &rhs) {
		/* Returns new matrix as the result of element-wise subtraction */
		if ((n != rhs.n) || (m != rhs.m)) {
			throw UnequalDim();
		}
		// create output matrix, initailized to zeros
		Matrix out(m, n);
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				out[i][j] = data[i][j] - rhs[i][j];
			}
		}

		return out;
	}
	Matrix operator*(Matrix &rhs) {
		/* Returns new matrix based on matrix multiplication 
		This means that for A * B = C, the element c at position i,j is the dot-product of column i and row j */
		// verify that the matrices are the right size
		if (n != rhs.m) {
			throw IncorrectDim();
		}
		// create our output matrix (initialized to zero)
		Matrix out(m, rhs.n);
		// we want to do our multiplications in parallel, to avoid race conditions
		// the calculations will be based on the output location
		// for loops could be merged but would require large CPU for utility 
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < rhs.n; j++) {
				for (int a = 0; a < n; a++) {
					out[i][j] += data[i][a] * rhs[a][j];
				}
			}
		}

		return out;
	}
	Matrix operator+(double &rhs) {
		/* This function takes in another matrix, sums this and the rhs element-wise, and returns a new matrix*/
		// create output matrix, initailized to zeros
		Matrix out(m, n);
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				out[i][j] = data[i][j] + rhs;
			}
		}

		return out;
	}
	Matrix operator-(double rhs) {
		/* Returns new matrix as the result of element-wise subtraction */
		// create output matrix, initailized to zeros
		Matrix out(m, n);
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				out[i][j] = data[i][j] - rhs;
			}
		}

		return out;
	}
	Matrix operator*(double rhs) {
		/* Returns new matrix based on scalar multiplication */
		// create output matrix, initailized to zeros
		Matrix out(m, n);
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				out[i][j] = data[i][j] * rhs;
			}
		}

		return out;
	}
	bool operator==(Matrix &rhs) {
		/* Test for equality */
		if (n != rhs.n) {
			return false;
		}
		if (m != rhs.m) {
			return false;
		}
		if (data != rhs.data) {
			return false;
		}
		return true;
	}
	bool operator!=(Matrix &rhs) {
		/* Test for inequality */
		return !(*this == rhs);
	}
	void operator=(const Matrix& rhs) {
		/* Explictly define the assignment operator (cpp probably would infer the exact same thing) */
		m = rhs.m;
		n = rhs.n;
		data = rhs.data;
	}


	// Elementary Row Operations
	void RowSwap(int row1, int row2) {
		/* Elementary row operation to swap rows*/
		data[row1].swap(data[row2]);
	}
	void RowAdd(int row1, int row2, double factor) {
		/* Elementary row operation to multiply row one by a factor and
		add the result to row two */
		#pragma omp parallel for
		for (int i = 0; i < n; i++) {
			data[row2][i] += factor * data[row1][i];
		}
	}
	void RowMul(int row, double amount) {
		/* Elementary row operation to multiply a row by a scalar */
		#pragma omp parallel for
		for (int i = 0; i < n; i++) {
			data[row][i] *= amount;
		}
	}


	// Other Basic Linear Algebra Functions
	Matrix T() {
		/* Function to calculate the transpose of this matrix */
		// This could be optimized and just flip a flag and the indexing order
		// but we aren't too worried about the data reads here yet
		Matrix out(n, m);
		#pragma omp parallel for
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				out[i][j] = data[j][i];
			}
		}
		return out;
	}
	double Det() {
		/* Function to calculate the determinant of the matrix 
		The determininat of a matrix is only defined if the matrix is square.
		Additionally, the determinant of a matrix is zeros if rank A < dim A. 
		The determinant of an upper triangular matrix is the product of its diagonal,
		so we will convert our matrix into this form.
		Since the Det(AB) = Det(A) * Det(B), we can use the elementray row operations
		in the first pass of Reduced Row Echelon Form (RREF), also called guassian elimination,
		to create this easier evaluation of the determinant. We would end up with
		Det(A) = Det(E_1 * E_2 * E_3 ... * A') = Det(E_1) * ... * Det(A'). 
			
		Det. of Elem. Row Swap Matrix = -1
		Det. of Elem. Row Addition Matrix = 1
		Det. of Elem. Row Multiplication Matrix = constant used for multiplication		
		Source: https://www.statlect.com/matrix-algebra/elementary-matrix-determinant
		*/
		// Verify we are a square matrix
		if (n != m) {
			throw NotSquare();
		}
		// Checking to see if any column contains only zeros
		// if so, then return 0
		double col_sums = 0;
		for (int j = 0; j < n; j++) {
			double col_sums = 0;
			for (int i = 0; i < m; i++) {
				col_sums += data[i][j];
			}
			if (col_sums <= 1e-16) {
				return 0;
			}
		}
		// Have ruled out easily checked trivial cases
		double det = 1;
		Matrix temp = *this;
		/* We will increment the rows as we convert the bottom triangle into
		zeros to keep track of level. We need a non-zero entry on each of the 
		leading terms along the diagonal (unless rankA < dimA) */
		for (int current_row = 0; current_row < m; current_row++) {
			double front = temp[current_row][current_row];
			// if the leading term is zero, we will need to do a row swap
			if (front <= 1e-16) {
				int i = current_row;
				while ((i < m) && (temp[i][current_row] <= 1e-16)) {
					i++;
				}
				if (i != current_row) {
					// update our data format so that we have swapped these rows
					temp.RowSwap(current_row, i);
					// update the value of our determinant
					det *= -1;
				}
			}
			// get new front
			front = temp[current_row][current_row];
			// now we want to use row addition of the current row
			// by a factor to make the columns below set to zero
			double other, factor;
			for (int i = current_row + 1; i < m; i++) {
				other = temp[i][current_row];
				factor = other / front;
				temp.RowAdd(current_row, i, factor);
				// det. of row addition is 1 (multiplicative identity)
			}
		}
		// we now have an upper-triangular matrix
		// to finish our calculation, we multiply the diagonal entries
		for (int i = 0; i < m; i++) {
			det *= temp[i][i];
		}

		return det;
	}
};

// Functions for the creation of matrices
Matrix Diag(std::vector<double> diag_entries) {
	/* Create a diagonal matrix with diagonal entries provided in a vector */
	// get size of vector
	int dim = diag_entries.size();
	// if vector is non-positive, throw error
	if (dim < 1) {
		throw ZeroRows();
	}
	Matrix out(dim, dim);
	for (int i = 0; i < dim; i++) {
		out[i][i] = diag_entries[i];
	}
	return out;
}

Matrix Eye(int n) {
	/* Create an identity matrix of size n */
	// check that n is positive
	if (n < 1) {
		throw ZeroRows();
	}
	std::vector<double> diag_entries(n, 1);
	return Diag(diag_entries);
}

// Linear Algebra Functions using Matrices
// These are declared outside of the Matrix class declaration to simplify the writing of our algorithms
Matrix Solve(Matrix A, Matrix s) {
	/* Solve a system of equations based on matrix of equations and vector of solutions
	Given Ax = s we want to find the vector x */
}

}
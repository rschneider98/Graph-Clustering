#pragma once
/* Richard Schneider
   24 JAN 2021
   Matrix Structure and Algorithms for Spectral Graph Clustering 
   FOR EDUCATIONAL PURPOSES, this is not designed or tested for use
   beyond algorithm implementation. Large scale and sparse matrices
   are also not considered since most operations produce new matrices
   and everything is stored as a dense matrix. */

#include <iostream>
#include <algorithm>
#include <cmath>
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

class UnderDetermined : public std::exception {
	virtual const char* what() const throw() {
		return "In order to be solved numerically, the matrix must have more rows than columns";
	}
};


class Matrix;
Matrix Diag(std::vector<double> diag_entries);
Matrix Eye(int n);
double InnerProduct(std::vector<double> u, std::vector<double> v);
std::vector<double> proj(std::vector<double> u, std::vector<double> v);
std::vector<double> operator+(std::vector<double> u, std::vector<double> v);
std::vector<double> operator-(std::vector<double> u, std::vector<double> v);
std::vector<double> operator*(std::vector<double> u, double c);


// Matrix class since that is the base data structure of Linear Algebra
class Matrix {
protected:
	// vector of vectors of doubles to represent the matrix in row-major form
	std::vector<std::vector<double>> data;
	// dimension of the matrix
	int m, n;

public:
	// empty constructor
	Matrix() {
		m = 0; n = 0;
	}
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

	Matrix Augment(Matrix& rhs) {
		/* This function takes two matrices and returns one concat. */
		// Verify they have the same number of rows
		if (m != rhs.m) { throw UnevenCols(); }
		int new_n = n + rhs.n;
		Matrix out = *this;
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = n; j < new_n; j++) {
				out[i][j] = rhs[i][j - n];
			}
		}
		return out;
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
				factor = -1 * (other / front);
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
	Matrix RREF(int aug = 0) {
		/* This function is meant to find the reduced row echolon form of 
		a matrix and return the new matrix. This is similar to the determinant
		function, but they cannot be combined because of restrictions on 
		square matrices and how the elementary functions affect the 
		calculation of the determinant. */
		// get number of columns to consider in matrix (if the matrix is augmented or not)
		// this defaults to 0 columns in the augmented section
		int num_rows = m;
		int num_cols = n - aug;
		int num_iter = std::min(num_rows, num_cols);
		// copy our matrix to get the output matrix that we will manipulate
		Matrix out = *this;
		/* We will increment the rows as we convert the bottom triangle into
		zeros to keep track of level. */
		for (int current_row = 0; current_row < num_iter; current_row++) {
			double front = out[current_row][current_row];
			// if the leading term is zero, we will need to do a row swap
			if (front <= 1e-16) {
				int i = current_row;
				while ((i < m) && (out[i][current_row] <= 1e-16)) {
					i++;
				}
				if ((i != current_row) && (out[current_row][i] > 1e-16)) {
					// update our data format so that we have swapped these rows
					out.RowSwap(current_row, i);
				}
			}
			// get new front
			front = out[current_row][current_row];
			// make sure that we did find a non zero entry for this row, if
			// not then we can skip this algebra bit
			if (front > 1e-16) {
				// use row multiplication to reduce this initial value to one
				out.RowMul(current_row, (1 / front));
				// now we want to use row addition of the current row
				// to make the columns below set to zero
				double other;
				for (int i = current_row + 1; i < m; i++) {
					other = out[i][current_row];
					out.RowAdd(current_row, i, -1 * other);
				}
			}
		}
		// we now have an upper-triangular matrix and need to 
		// remove values above the diagonal
		for (int current_row = num_iter; current_row > 0; current_row--) {
			double front = out[current_row][current_row];
			// if this value is nonzero
			if (front > 1e-16) {
				for (int i = 0; i < current_row; i++) {
					out.RowAdd(current_row, i, -1 * out[i][current_row]);
				}
			}
		}
		return out;
	}
	std::vector<double> Solve(std::vector<double> x) {
		/* Solve the matrix for solutions x */
		// Make sure that there is probably enough rows to solve the problem
		if (m < n) { throw UnderDetermined(); }
		// Verify the size of x is equivalent to the number of rows
		if (m != x.size()) { throw UnevenCols(); }
		// Create an augmented matrix 
		Matrix temp_vector = Matrix(m, 1);
		for (int i = 0; i < m; i++) {
			temp_vector[i][0] = x[i];
		}
		Matrix comp = Augment(temp_vector);
		Matrix solve = comp.RREF(1);
		// Verify that we have a solution, meaning the diagonal
		// entries are all zeros
		for (int i = 0; i < n; i++) {
			if (std::abs(solve[i][i]) <= 1e-16) { throw UnderDetermined(); }
		}
		std::vector<double> solutions(m, 0);
		for (int i = 0; i < n; i++) {
			solutions[i] = solve[m][i];
		}

		return solutions;
	}
	Matrix GramSchmidt() {
		/* Here we apply the Gram-Schmidt process to this matrix to 
		find an orthogonal basis. There are a couple of functions needed
		to compute this, specifically the inner product of two vectors
		and the projection of two vectors. */


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


// Helper Functions for Vectors
double InnerProduct(std::vector<double> u, std::vector<double> v) {
	/* This computes the inner product of two vectors (also known as
	the dot product) */
	if (u.size() != v.size()) { throw UnequalDim(); }
	int len = u.size();
	double out;
	for (int i = 0; i < len; i++) {
		out += u[i] * v[i];
	}
	return out;
}

std::vector<double> proj(std::vector<double> u, std::vector<double> v) {
	/* This function finds the projection of v onto u 
	proj_u (v) = \frac{<u, v>}{<u, u>}u
	*/
	if (u.size() != v.size()) { throw UnequalDim(); }
	double num = InnerProduct(u, v);
	double denom = InnerProduct(u, u);
	if (std::abs(denom) <= 1e-16) { return std::vector<double> (v.size(), 0); }
	return u * (num / denom);
}

std::vector<double> operator+(std::vector<double> u, std::vector<double> v) {
	/* This function is vector addition performed element-wise */
	// Verify both vectors are the same length
	if (u.size() != v.size()) { throw UnequalDim(); }
	std::vector<double> out(u.size(), 0);
	#pragma omp parallel for 
	for (int i = 0; i < u.size(); i++) {
		out[i] = u[i] + v[i];
	}
	return out;
}

std::vector<double> operator-(std::vector<double> u, std::vector<double> v) {
	/* This function is vector subtraction performed element-wise */
	// Verify both vectors are the same length
	if (u.size() != v.size()) { throw UnequalDim(); }
	std::vector<double> out(u.size(), 0);
	#pragma omp parallel for 
	for (int i = 0; i < u.size(); i++) {
		out[i] = u[i] - v[i];
	}
	return out;
}

std::vector<double> operator*(std::vector<double> u, double c) {
	/* This function is vector scalar multiplication performed element-wise */
	std::vector<double> out = u;
	#pragma omp parallel for 
	for (int i = 0; i < u.size(); i++) {
		out[i] *= c;
	}
	return out;
}



// Linear Algebra Functions using Matrices
// These are declared outside of the Matrix class declaration to simplify the writing of our algorithms


}
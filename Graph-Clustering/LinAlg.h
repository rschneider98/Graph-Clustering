#pragma once
/* Richard Schneider
   24 JAN 2021
   Matrix Structure and Algorithms for Spectral Graph Clustering 
   FOR EDUCATIONAL PURPOSES, this is not designed or tested for use
   beyond algorithm implementation. Large scale and sparse matrices
   are also not considered since most operations produce new matrices
   and everything is stored as a dense matrix. */

#include <iostream>
#include <exception>
#include <algorithm>
#include <utility> 
#include <cmath>
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
class Vector;
Matrix Diag(std::vector<double> diag_entries);
Matrix Eye(int n);


// Create a Vector class that is just a wrapper for std::vector with 
// so overloaded operators don't affect other operations
class Vector {
protected:
	std::vector<double> data;
public:
	// constructors
	Vector() {};
	Vector(int len, double fill) {
		data = std::vector<double>(len, fill);
	}
	Vector(std::vector<double> my_vect) {
		data = my_vect;
	}

	// Operator Overloading
	double& operator[](int i) {
		/* This function returns the address of the double at index.
		so this can be used to update and get data. */
		return data[i];
	}
	Vector operator+(Vector rhs) {
		/* This function is vector addition performed element-wise */
		// Verify both vectors are the same length
		if (data.size() != rhs.data.size()) { throw UnequalDim(); }
		Vector out(data.size(), 0);
#pragma omp parallel for 
		for (int i = 0; i < data.size(); i++) {
			out[i] = data[i] + rhs[i];
		}
		return out;
	}
	Vector operator-(Vector rhs) {
		/* This function is vector subtraction performed element-wise */
		// Verify both vectors are the same length
		if (data.size() != rhs.data.size()) { throw UnequalDim(); }
		Vector out(data.size(), 0);
#pragma omp parallel for 
		for (int i = 0; i < data.size(); i++) {
			out[i] = data[i] - rhs[i];
		}
		return out;
	}
	Vector operator*(double c) {
		/* This function is vector scalar multiplication performed element-wise */
		Vector out(*this);
#pragma omp parallel for 
		for (int i = 0; i < data.size(); i++) {
			out[i] *= c;
		}
		return out;
	}
	Vector operator=(Vector rhs) {
		/* This function is vector scalar multiplication performed element-wise */
		Vector out(rhs.data);
		return out;
	}
	double mag() {
		/* Returns the magnitude of the vector, also known as the absolute value or length */
		double out = 0;
		for (int i = 0; i < data.size(); i++) {
			out += std::pow(data[i], 2);
		}
		return std::sqrt(out);
	}

	// Helper Functions for Vectors
	int size() {
		return data.size();
	}
	double InnerProduct(Vector rhs) {
		/* This computes the inner product of two vectors (also known as
		the dot product) */
		if (data.size() != rhs.data.size()) { throw UnequalDim(); }
		int len = data.size();
		double out = 0;
		for (int i = 0; i < len; i++) {
			out += data[i] * rhs[i];
		}
		return out;
	}
	Vector proj(Vector rhs) {
		/* This function finds the projection of v onto u
		proj_u (v) = \frac{<u, v>}{<u, u>}u
		*/
		if (data.size() != rhs.data.size()) { throw UnequalDim(); }
		double num = (*this).InnerProduct(rhs);
		double denom = (*this).InnerProduct((*this));
		if (std::abs(denom) <= 1e-16) { return std::vector<double>(rhs.data.size(), 0); }
		return (*this) * (num / denom);
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
	// constructer based on merging column vectors
	Matrix(std::vector<Vector> cols) {
		// get number of columns
		n = cols.size();
		// verify there is at least one column
		if (n == 0) { throw ZeroCols(); }
		// get number of rows in first column
		m = cols[0].size();
		// verify that there is at least one row
		if (m == 0) { throw ZeroRows(); }
		// verify that all columns have same number of rows
		for (int i = 0; i < n; i++) {
			if (cols[i].size() != m) { throw UnevenCols(); }
		}
		// create our data structure
		data = std::vector<std::vector<double>>(m, (std::vector<double>(n, 0)));
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				data[i][j] = cols[j][i];
			}
		}
	}
		

	// data access operator overloading
	std::vector<double>& operator[](const int i) {
		/* This overloads the access operator to only use one bracket instead of two, and it
		automatically references the data vectors */
		return data[i];
	}
	double& at(const int i, const int j) {
		/* Returns tha data at location i, j */
		return data[i][j];
	}


	// arithmetic operator overloading
	Matrix operator+(const Matrix &rhs) {
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
				out[i][j] = data[i][j] + rhs.data[i][j];
			}
		}

		return out;
	}
	Matrix operator-(const Matrix &rhs) {
		/* Returns new matrix as the result of element-wise subtraction */
		if ((n != rhs.n) || (m != rhs.m)) {
			throw UnequalDim();
		}
		// create output matrix, initailized to zeros
		Matrix out(m, n);
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				out[i][j] = data[i][j] - rhs.data[i][j];
			}
		}

		return out;
	}
	Matrix operator*(const Matrix &rhs) {
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
					out[i][j] += data[i][a] * rhs.data[a][j];
				}
			}
		}

		return out;
	}
	Matrix operator+(const double &rhs) {
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
	Matrix operator-(const double rhs) {
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
	Matrix operator*(const double rhs) {
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
	bool operator==(const Matrix &rhs) {
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
	bool operator!=(const Matrix &rhs) {
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
	Vector GetCol(int col_num) {
		/* This function creates a vector based on the column number */
		// Output Vector
		Vector out(m, 0);
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			out[i] = data[i][col_num];
		}
		return out;
	}
	Vector GetRow(int row_num) {
		/* This function returns a vector created from a row */
		Vector out(data[row_num]);
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
			if (std::abs(col_sums) <= 1e-16) {
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
			if (std::abs(front) <= 1e-16) {
				int i = current_row;
				while ((i < m) && (std::abs(temp[i][current_row]) <= 1e-16)) {
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
		Matrix out(data);
		/* We will increment the rows as we convert the bottom triangle into
		zeros to keep track of level. */
		for (int current_row = 0; current_row < num_iter; current_row++) {
			double front = out[current_row][current_row];
			// if the leading term is zero, we will need to do a row swap
			if (std::abs(front) <= 1e-16) {
				int i = current_row;
				while ((i < m) && (std::abs(out[i][current_row]) <= 1e-16)) {
					i++;
				}
				if ((i != current_row) && (std::abs(out[current_row][i]) > 1e-16)) {
					// update our data format so that we have swapped these rows
					out.RowSwap(current_row, i);
				}
			}
			// get new front
			front = out[current_row][current_row];
			// make sure that we did find a non zero entry for this row, if
			// not then we can skip this algebra bit
			if (std::abs(front) > 1e-16) {
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
			if (std::abs(front) > 1e-16) {
				for (int i = 0; i < current_row; i++) {
					out.RowAdd(current_row, i, -1 * out[i][current_row]);
				}
			}
		}
		return out;
	}
	Vector Solve(Vector x) {
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
		Vector solutions(m, 0);
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
		// the sequence of u's
		std::vector<Vector> U(n, Vector(m, 0));
		// set the first u to itself
		Vector v = (*this).GetCol(0);
		U[0] = v * (1 / v.mag());
		for (int k = 1; k < n; k++) {
			// get the next column vector
			Vector v = (*this).GetCol(k);
			Vector u = v;
			for (int i = (k - 1); k >= 0; k--) {
				u = u - U[i].proj(v);
			}
			U[k] = u * (1 / u.mag());
		}

		return Matrix(U);
	}
	std::pair<Matrix, Matrix> QRDecomp() {
		/* This function takes this matrix and performs QR decomposition. 
		Q is the orthonormal basis, which can be found by the Gram-Schmidt process
		This is orthogonal, so its inverse is also the transpose.
		A = QR -> Q^T * A = R
		R is then also an upper-triangular matrix. */
		Matrix Q = (*this).GramSchmidt();
		Matrix R = Q.T() * (*this);
		return std::make_pair(Q, R);
	}
	bool isTrig() {
		/* Bool of a matrix to test if the matrix is upper-triangular */
		bool out = true;
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				// if below our main diagonal, check if entries are 0
				if (i < j) {
					if (std::abs(data[i][j]) > 1e-16) {
						out = false;
					}
				}
			}
		}
		return out;
	}
	std::vector<double> Eigenvalues() {
		/* This uses the QR-algorithm to calculate the eigenvalues of the matrix
		This is done my repeatedly finding A_k = Q_k * R_k  and then 
		A_{k+1} = R_{k} * Q_{k} until A_{k+1} is upper-triangular. Then 
		the eigenvalues are those found on the diagonal. There is a proof of this. 
		Note: This process can be speed up with Householder reductions first to put the 
		matrix in Hessenberg form since it is closer to convergence. NOT done here */
		std::pair<Matrix, Matrix> QR_pair = (*this).QRDecomp();
		Matrix temp = QR_pair.second * QR_pair.first;
		while (!temp.isTrig()) {
			QR_pair = temp.QRDecomp();
			temp = QR_pair.second * QR_pair.first;
		}
		std::vector<double> out(n, 0);
		for (int i = 0; i < n; i++) {
			out[i] = temp[i][i];
		}
		return out;
	}
	Vector Eigenvector(double eigenvalue) {
		/* This takes a eigenvalue, and solves for the eigenvectors 
		Want (A - eigenvalue * I) * x = 0 */
		Matrix temp = (*this) - (Eye(n) * eigenvalue);
		Vector solution = temp.Solve(Vector(n, 0));
		return solution;
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


}
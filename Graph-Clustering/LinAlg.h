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
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <utility> 
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
	void operator=(Vector rhs) {
		/* This function is vector scalar multiplication performed element-wise */
		data =rhs.data;
	}
	bool operator==(Vector& rhs) {
		/* This function evaluates the equality of two vectors */
		int s = data.size();
		if (s != rhs.data.size()) {
			return false;
		}
		for (int i = 0; i < s; i++) {
			if (std::abs(data[i] - rhs.data[i]) > 1e-6) {
				return false;
			}
		}
		return true;
	}
	friend bool operator==(const Vector& lhs, const Vector& rhs) {
		/* This function evaluates the equality of two vectors */
		int s = lhs.data.size();
		if (s != rhs.data.size()) {
			return false;
		}
		for (int i = 0; i < s; i++) {
			if (std::abs(lhs.data[i] - rhs.data[i]) > 1e-6) {
				return false;
			}
		}
		return true;
	}
	friend bool operator!=(const Vector& lhs, const Vector& rhs) {
		/* This function evaluates the inequality of two vectors */
		return !(lhs == rhs);
	}
	friend std::ostream& operator<<(std::ostream& out, Vector& rhs) {
		/* Print out the vector */
		out << "[ ";
		for (int i = 0; i < rhs.data.size(); i++) {
			out << std::to_string(rhs.data[i]) << " ";
		}
		out << "]" << std::endl;
		return out;
	}
	void toFile(std::string fname) {
		std::ofstream outfile;
		outfile.open(fname, std::ios::out);
		if (outfile.is_open()) {
			// read lines
			for (int i = 0; i < data.size(); i++) {
				outfile << std::to_string(data[i]) << std::endl;
			}
			outfile.close();
		}
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
		u = this
		v = rhs
		*/
		if (data.size() != rhs.data.size()) { throw UnequalDim(); }
		double num = (*this).InnerProduct(rhs);
		double denom = (*this).InnerProduct((*this));
		if (std::abs(denom) <= 1e-6) { return std::vector<double>(rhs.data.size(), 0); }
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
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (std::abs(data[i][j] - data[i][j]) > 1e-6) {
					return false;
				}
			}
		}
		return true;
	}
	friend bool operator==(const Matrix& lhs, const Matrix& rhs) {
		/* Test for equality */
		if (lhs.n != rhs.n) {
			return false;
		}
		if (lhs.m != rhs.m) {
			return false;
		}
		for (int i = 0; i < lhs.m; i++) {
			for (int j = 0; j < lhs.n; j++) {
				if (std::abs(lhs.data[i][j] - rhs.data[i][j]) > 1e-6) {
					return false;
				}
			}
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
	friend std::ostream& operator<<(std::ostream& out, Matrix &rhs) {
		/* Print out the matrix */
		for (int i = 0; i < rhs.m; i++) {
			for (int j = 0; j < rhs.n; j++) {
				out << std::to_string(rhs.data[i][j]) << " ";
			}
			out << std::endl;
		}
		return out;
	}
	void toFile(std::string fname) {
		std::ofstream outfile;
		outfile.open(fname, std::ios::out);
		if (outfile.is_open()) {
			// write to a temp stringstream and count number of nonzero elements
			int num_elem = 0;
			std::stringstream temp;
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					if (std::abs(data[i][j]) > 1e-6) {
						num_elem++;
						temp << std::to_string(i) << " ";
						temp << std::to_string(j) << " ";
						temp << std::to_string(data[i][j]) << std::endl;
					}
				}
			}
			// output header of Matrix Market file
			outfile << "%%MatrixMarket matrix coordinate real general"
				"%= ================================================================================\n"
				"%\n"
				"%This ASCII file represents a sparse MxN matrix with L\n"
				"% nonzeros in the following Matrix Market format :\n";
			outfile << std::to_string(m) << " " << std::to_string(n) << " " << std::to_string(num_elem) << std::endl;
			outfile << temp.rdbuf();
			outfile.close();
		}
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
		int new_m = m;
		int new_n = n + rhs.n;
		Matrix out(new_m, new_n);
#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				out[i][j] = data[i][j];
			}
		}
#pragma omp parallel for
		for (int i = 0; i < rhs.m; i++) {
			for (int j = 0; j < rhs.n; j++) {
				out[i][n + j] = rhs.data[i][j];
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
	Matrix RREF() {
		/* This function is meant to find the reduced row echolon form of 
		a matrix and return the new matrix. This is similar to the determinant
		function, but they cannot be combined because of restrictions on 
		square matrices and how the elementary functions affect the 
		calculation of the determinant. */
		// get number of columns to consider in matrix (if the matrix is augmented or not)
		// this defaults to 0 columns in the augmented section
		int num_rows = m;
		int num_cols = n;
		int num_iter = std::min(num_rows, num_cols);
		// copy our matrix to get the output matrix that we will manipulate
		Matrix out = (*this);
		/* We will increment the rows as we convert the bottom triangle into
		zeros to keep track of level. */
		for (int current_row = 0; current_row < num_iter; current_row++) {
			double front = out[current_row][current_row];
			// if the leading term is zero, we will need to do a row swap
			if (std::abs(front) <= 1e-6) {
				int i = current_row;
				while ((i < (m - 1)) && (std::abs(out[i][current_row]) <= 1e-6)) {
					i++;
				}
				if ((i != current_row) && (std::abs(out[i][current_row]) > 1e-6)) {
					// update our data format so that we have swapped these rows
					out.RowSwap(current_row, i);
				}
			}
			// get new front
			front = out[current_row][current_row];
			// make sure that we did find a non zero entry for this row, if
			// not then we can skip this algebra bit
			if (std::abs(front) > 1e-6) {
				// use row multiplication to reduce this initial value to one
				out.RowMul(current_row, (1.0 / front));
				// now we want to use row addition of the current row
				// to make the columns below set to zero
				double other;
				for (int i = current_row + 1; i < m; i++) {
					other = out[i][current_row];
					out.RowAdd(current_row, i, -1.0 * other);
				}
			}
		}
		// need to pick up the rest of the columns
		if (std::abs(out[num_iter - 1][num_iter - 1] - 1) > 1e-6) {
			// this will only occur if cols > rows
			int j = num_iter - 1;
			bool escape = false;
			while ((j < n) && (!escape)) {
				int front = out[num_iter - 1][j];
				if (std::abs(front) > 1e-6) { 
					out.RowMul(num_iter - 1, (1.0 / front));
					escape = true;
				}
				else {
					j++;
				}
			}
			
		}
		// we now have an upper-triangular matrix and need to 
		// remove values above the diagonal
		// pick up the extra columns that are past the min(num rows, num cols)
		int i = m - 1;
		int j = n - 1;
		while ((j >= num_iter)) {
			if (std::abs(out[i][j]) > 1e-6) {
				for (int r = 0; r < i; r++) {
					out.RowAdd(i, r, -1 * out[r][j]);
				}
			}
			j--;
		}
		// after we are back to the square part, our algorithm should have 1's
		// along the diagonal
		for (int current_row = (num_iter - 1); current_row >= 0; current_row--) {
			double front = out[current_row][current_row];
			// if this value is nonzero
			if (std::abs(front) > 1e-6) {
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
		Matrix comp = (*this).Augment(temp_vector);
		Matrix solve = comp.RREF();
		// Verify that we have a solution, meaning the diagonal
		// entries are all nonzero
		for (int i = 0; i < n; i++) {
			if (std::abs(solve[i][i]) <= 1e-16) { throw UnderDetermined(); }
		}
		Vector solutions(m, 0);
		for (int i = 0; i < m; i++) {
			solutions[i] = solve[i][n];
		}

		return solutions;
	}
	Matrix GramSchmidt() {
		/* Here we apply the Gram-Schmidt process to this matrix to 
		find an orthogonal basis. There are a couple of functions needed
		to compute this, specifically the inner product of two vectors
		and the projection of two vectors. */
		// the sequence of u's
		std::vector<Vector> U, E;
		// set the first u to itself
		Vector v = (*this).GetCol(0);
		Vector u = v;
		U.push_back(u);
		E.push_back(u * (1.0 / u.mag()));
		for (int k = 1; k < n; k++) {
			// get the next column vector
			Vector v = (*this).GetCol(k);
			Vector u = v;
			for (int i = (k - 1); i >= 0; i--) {
				u = u - U[i].proj(v);
			}
			U.push_back(u);
			E.push_back(u * (1.0 / u.mag()));
		}

		return Matrix(E);
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
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				// if below our main diagonal, check if entries are 0
				if (i > j) {
					if (std::abs(data[i][j]) > 1e-6) {
						out = false;
					}
				}
			}
		}
		return out;
	}
	bool isComplexTrig() {
		/* Bool of a SQUARE matrix to test if the matrix is upper-triangular with only
		imaginary conjugates of eigenvalues below the diagonal. These can
		only occur every other step */
		// check one step below main diagonal
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				// if below our main diagonal by one row, check if entries are 0
				if ((i - 1) > j) {
					if (std::abs(data[i][j]) > 1e-6) {
						return false;
					}
				}
			}
		}
		// check row below main diagonal
		double prev = data[1][0];
		double next;
		for (int i = 1; i < (m - 1); i++) {
			next = data[i + 1][i];
			if ((std::abs(prev) > 1e-6) && (std::abs(next) > 1e-6)) {
				return false;
			}
		}
		return true;
	}
	Matrix EigenvaluesTwoByTwo() {
		/* Find eigenvalues of a 2x2 matrix and return matrix in complex form
		This equation is 
		\lambda^2 - (a_{11} + a_{22}) \lambda + a_[11} * a_{22} -a_{12} * a_{21} = 0 */
		Matrix out(2, 2);
		double a = 1;
		double b = -1 * (data[0][0] + data[1][1]);
		double c = (data[0][0] * data[1][1]) - (data[0][1] * data[1][0]);
		double root = std::pow(b, 2) - (4 * a * c);
		if (root < 0) {
			double real = (-1 * b) / (2 * a);
			double img = std::sqrt(-1 * root) / ( 2 * a);
			out[0][0] = real;
			out[1][1] = real;
			out[0][1] = img;
			out[1][0] = -1 * img;
		}
		else {
			out[0][0] = (-1 * b + std::sqrt(root)) / (2 * a);
			out[1][1] = (-1 * b - std::sqrt(root)) / (2 * a);
		}
		return out;
	}
	Matrix getFinalEigenvalues() {
		/* This functions takes the current matrix in complex upper 
		triangular form, looks for complex eigenvalues along
		the main diagonal and returns a matrix with
		calculated complex eigenvalues */
		Matrix out = (*this);
		// along the diagonal below the main diag
		for (int i = 0; i < (m - 1); i++) {
			// if nonzero, this is the imaginary component of 
			// the eigenvector above
			if (std::abs(data[i + 1][i]) > 1e-6) {
				// create temp matrix of these complex components
				Matrix temp(2, 2);
				temp[0][0] = data[i][i];
				temp[0][1] = data[i][i + 1];
				temp[1][0] = data[i + 1][i];
				temp[1][1] = data[i + 1][i + 1];
				Matrix solution = temp.EigenvaluesTwoByTwo();
				out[i][i] = solution[0][0];
				out[i][i + 1] = solution[0][1];
				out[i + 1][i] = solution[1][0];
				out[i + 1][i + 1] = solution[1][1];
			}
		}
		return out;
	}
	std::pair<Matrix, double> vhouse() {
		/* Construction of a Householder vector 
		
		https://math.la.asu.edu/~gardner/QR.pdf
		H = I - 2uuT/||u||^2
		Want a v and beta such that Hv = [||r|| 0 0 0 ...]
		So v = x - r and beta = 2/||v||^2*/
		Matrix v = (*this);
		v[0][0] -= (*this).GetCol(0).mag();
		double beta = 2 / std::pow(v.GetCol(0).mag(), 2);
		return std::make_pair(v, beta);
	}
	std::pair<Matrix, Matrix> toHessenberg() {
		/* This will transform a matrix to similar one in Hessenberg form 
		http://webhome.auburn.edu/~tamtiny/lecture%2010.pdf
		Compute a similar matrix in Hessenberg form using householder transformations

		Not used: http://www.foo.be/docs-free/Numerical_Recipe_In_C/c11-5.pdf 
		This is a modified version of guassian elimination in order to
		form a matrix that is Hessenberg form and similar */
		// verify we have a square matrix
		if (m != n) { throw NotSquare(); }
		Matrix Q = Eye(m);
		Matrix H = (*this);
		for (int k = 1; k < (n - 1); k++) {
			// create a limited vector of entries below main diagonal
			Matrix v(m - k, 1);
			for (int i = 0; i < (m - k); i++) {
				v[i][0] = H[i+k][k - 1];
			}

			// create householder matrix for this limited vector of matrix
			// P_k = I - 2uu^T when u is a unit vector
			// find R (lower right matrix)
			std::pair<Matrix, double> hvect = v.vhouse();
			Matrix u = hvect.first;
			double beta = hvect.second;
			// compute householder matrix
			Matrix R = Eye(u.m) - (u * u.T() * beta);
			
			// need to make new matrix P
			// P = [I(k), zeros(k, n-k); zeros(n-k, k), R]
			Matrix P(m, m);
			for (int i = 0; i < k; i++) {
				P[i][i] = 1;
			}
#pragma omp parallel for
			for (int i = 0; i < v.m; i++) {
				for (int j = 0; j < v.m; j++) {
					P[i + k][j + k] = R[i][j];
				}
			}
			Q = Q * P;
			H = P * H;
			H = H * P.T(); 
			
		}
		// after looping through all columns, we return H and Q
		return std::make_pair(H, Q);
	}
	std::pair<Matrix, Matrix> QRAlgorithm() {
		/* This uses the QR-algorithm to calculate the eigenvalues and eigenvectors of the matrix
		
		This is done my repeatedly finding A_k = Q_k * R_k  and then 
		A_{k+1} = R_{k} * Q_{k} until A_{k+1} is upper-triangular. Then 
		the eigenvalues are those found on the diagonal. There is a proof of this. 

		Notes: See http://www.mosismath.com/Eigenvalues/EigenvalsQR.html for more
		details on implementaion and usage of the QR algorithm

		This process can be speed up with Householder reductions first to put the 
		matrix in Hessenberg form since it is closer to convergence. This is implemented 
		becuase matrices larger than 10 x 10 take a very long time to converge. 
		Without this, a random 20 x 20 matrix took 4:30 minutes; with this, it took ~2 seconds.

		For complex results, they are stored in 2 x 2 matrices along the diagonal
		[Re_1, Im_2]
		[Im_1, Re_2]
		This means that Im_1 is below the diagonal and our exit condition for real numbers
		is insufficient. Thus, we will store the entries of the diagonal, and if they 
		are equivalent, then we assume that we have reached our exit condition.
		
		Usage of "shifts" helps increase the speed of convergence
		H_n - cI = QR
		 H_{n+1} = RQ + cI 
		The last two diagnoal entries are used here for the shift value 
		
		Eigenvectors are the columns of the product of all of the Q matrices */
		// two by two matrix we calculate using characteristic 
		// polynomial and quadratic equation then return real parts
		if (m == n == 2) { 
			Matrix values = (*this).EigenvaluesTwoByTwo();
			std::pair<Matrix, Matrix> QR_pair = values.QRDecomp();
			return std::make_pair(values, QR_pair.first);
		}

		// convert to Hessenberg form first
		std::pair<Matrix, Matrix> Hess = (*this).toHessenberg();
		Matrix temp1 = Hess.first;
		Matrix eigenvects = Hess.second;

		// first QR decomposition
		std::pair<Matrix, Matrix> QR_pair = temp1.QRDecomp();
		temp1 = QR_pair.second * QR_pair.first;
		eigenvects = eigenvects * QR_pair.first;
		Matrix temp2(m, n);
		int escape = 0; // do at least 3 iterations for complex numbers so that they are stable
		int flip = 0; // used to designate whether to use last diagonal or second to last
		while ((!temp1.isTrig()) && (escape < 3)) {
			// shift the matrix
			double factor = temp1[m - 1 - flip][m - 1 - flip];
			flip = (flip + 1) % 3;
			temp2 = temp1 - (Eye(m) * factor);
			// get QR decomposition of our matrix
			QR_pair = temp2.QRDecomp();
			// find similar matrix closer to upper triangular
			temp2 = QR_pair.second * QR_pair.first;
			// update matrix of eigenvectors
			eigenvects = eigenvects * QR_pair.first;
			// unshift matrix
			temp1 = temp2 + (Eye(m) * factor);
			// if in complex form, use 2x2 eigenvalue calculations
			if (temp1.isComplexTrig()) {
				temp1 = temp1.getFinalEigenvalues();
				escape += 1;
			}
		}
		return std::make_pair(temp1, eigenvects);
	}

	std::vector<double> Eigenvalues() {
		/* This uses the QR-algorithm to calculate the REAL eigenvalues of the matrix */
		std::pair<Matrix, Matrix> eigenPair = (*this).QRAlgorithm();
		Matrix matrixValues = eigenPair.first;
		std::vector<double> out(n, 0);
		for (int i = 0; i < n; i++) {
			out[i] = matrixValues[i][i];
		}
		return out;
	}
	Matrix Eigenvectors() {
		/* This returns a matrix of eigenvectors for all eigenvalues */
		std::pair<Matrix, Matrix> eigenPair = (*this).QRAlgorithm();
		return eigenPair.second;
	}
	Vector KMeans(int k) {
		/* K-Means clustering of m, n-dimensional observations (each row) */
		int max_iter = 10;
		int num_ints = 0;

		// get random assignments
		Vector ass(m, 0);
		Vector prev_ass = ass;
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution(0, k - 1);
		for (int i = 0; i < m; i++) {
			ass[i] = distribution(generator);
		}
		// iteratively recalculate cluster centroids 
		Matrix centroid(k, n);
		while ((prev_ass != ass) && (num_ints < max_iter)) {
			// find the average of the assigned clusters
			Vector counts(k, 0);
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					centroid[ass[i]][j] += data[i][j];
				}
				counts[ass[i]] += 1;
			}
			for (int i = 0; i < k; i++) {
				for (int j = 0; j < n; j++) {
					centroid[i][j] *= (1 / counts[i]);
				}
			}
			// reassign clusters
			prev_ass = ass;
#pragma omp parallel for
			for (int i = 0; i < m; i++) {
				double temp;
				int loc = 0;
				Vector row = (*this).GetRow(i);
				double min = (centroid.GetRow(0) - row).mag();
				for (int j = 1; j < k; j++) {
					temp = (centroid.GetRow(j) - row).mag();
					if (temp < min) {
						min = temp;
						loc = j;
					}
				}
				ass[i] = loc;
			}
			// increment iteration counter
			num_ints++;
		}
		return ass;
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

}
#pragma once
// Richard Schneider
// 24 JAN 2021
// Matrix Structure and Algorithms for Spectral Graph Clustering

#include <iostream>
#include <exception>
#include <vector>

namespace LinAlg {

// exceptions used in this namespace
class ZeroRows : public exception {
	virtual const char* what() const throw() {
		return "A Matrix declared with a vector must have at least one row";
	}
};

class ZeroCols : public exception {
	virtual const char* what() const throw() {
		return "A Matrix declared with a vector must have at least one column";
	}
};

class UnevenCols : public exception {
	virtual const char* what() const throw() {
		return "A Matrix declared with a vector must have columns of equal sizes";
	}
};

class UnequalDim : public exception {
	virtual const char* what() const throw() {
		return "Matrices must be of equal dimensions for addition";
	}
};

class IncorrectDim : public exception {
	virtual const char* what() const throw() {
		return "For matrix multiplication, the number of rows for left-hand side matrix must be equal to the number of columns of the right-hand matrix";
	}
};


// Matrix class since that is the base data structure of Linear Algebra
class Matrix {
protected:
	// vector of vectors of doubles to represent the matrix in row-major form
	std::vector<std::vector<double>> data;
	// dimension of the matrix
	int m, n;

private:
	void RowSwap(int row1, int row2) {
		/* Elementary row operation to swap rows*/

	}
	void RowAdd(int row1, int row2) {
		/* Elementary row operation to add row one to row two */

	}
	void RowMul(int row, double amount) {
		/* Elementary row operation to multiply a row by a scalar */

	}

public:
	// empty constructor
	Matrix() {}
	// constructor with shape (defaults to zeros)
	Matrix(int m, int n) {
		this->m = m;
		this->n = n;
		data.assign(m, std::vector<int>(n, 0));
	}
	// constructor based on passing in the data
	Matrix(std::vector<std::vector<double>> data) {
		// get number of rows 
		m = data.size();
		// verify there is at least one row
		if (m == 0) {
			throw ZeroRows;
		}
		// get the number of columns for the first row
		n = data[0].size();
		// verify there is at least one column
		if (n == 0) {
			throw ZeroCols;
		}
		// verify each row has the same number of columns
		for (int i = 0; i < m; i++) {
			if (data[i].size() != n) {
				throw UnevenCols;
			}
		}
		// we have validated the input data and can keep it
		this->data = data;
	}

	// arithmetic operator overloading
	Matrix operator+(Matrix rhs) {
		/* This function takes in another matrix, sums this and the rhs element-wise, and returns a new matrix*/

	}
	Matrix operator-(Matrix rhs) {
		/* Returns new matrix as the result of element-wise subtraction */

	}
	Matrix operator*(Matrix rhs) {
		/* Returns new matrix based on matrix multiplication 
		This means that for A * B = C, the element c at position i,j is the dot-product of column i and row j */
		
	}
	Matrix operator+(double rhs) {
		/* This function takes in another matrix, sums this and the rhs element-wise, and returns a new matrix*/

	}
	Matrix operator-(double rhs) {
		/* Returns new matrix as the result of element-wise subtraction */

	}
	Matrix operator*(double rhs) {
		/* Returns new matrix based on scalar multiplication */

	}
	Matrix operator==(Matrix rhs) {
		/* Test for equality */

	}
	Matrix operator!=(Matrix rhs) {
		/* Test for inequality */

	}
	void operator=(const Matrix& rhs) {
		/* Explictly define the assignment operator (cpp probably would infer the exact same thing) */
		m = rhs.m;
		n = rhs.n;
		data = rhs.data;
	}

	// Other Basic Linear Algebra Functions
	Matrix transpose() {
		/* Function to calculate the transpose of this matrix */
	}
	double Determinant() {
		/* Function to calculate the determinant of the matrix */
	}
	Matrix Inverse() {
		/* Function to calculate the inverse of this matrix 
		Note: A matrix is only invertible if it is a square and the determinant is nonzero */
	}
};

// Functions for the creation of matrices
Matrix Diag(vector<double> DiagEntries) {
	/* Create a diagonal matrix with diagonal entries provided in a vector */

}

Matrix Eye(int n) {
	/* Create an identity matrix of size n */
}

// Linear Algebra Functions using Matrices
// These are declared outside of the Matrix class declaration to allow use of elemetary matrices
// This simplifies the writing of our algorithms
Matrix Solve(Matrix A, Matrix s) {
	/* Solve a system of equations based on matrix of equations and vector of solutions
	Given Ax = s we want to find the vector x */
}

}
# Eigenvector Clustering of Graph
This repo is a project that implements the eigenvector clustering algorithm for a graph. This was created for educational experience. As such, the linear algebra routines are not as optimized as possible given common numerical methods or libraries could be. Eigenvector clustering is used for graphs to represent the relational information as vectorized parameters so that the vertices of a network can be clustered. After this is applied, we can use any vectorized clustering algorithm (common ones that might appear in a statistics class). Here we used our own implementation of K-Means. 

We use a dataset of 494 vertices that represent points in the US power grid (http://networkrepository.com/power-494-bus.php, https://www.cise.ufl.edu/research/sparse/matrices/HB/494_bus). This is still considered a rather small graph and matrix, but useful for our goal of understanding this clustering method better.

## Eigenvector Clustering 
Given a graph $G$, we have an adjacency matrix $A$ and diagonal degree matrix $D$. The Laplacian matrix is then $L = D - A$. For clustering the first $x$ eigenvectors of this matrix corresponding to the $x$ smallest eigenvalues are calculated. This produces a $n \cross x$ matrix where each node's vector representation is the $i$th row. These vectors can be passed to a vectorized clustering algorithm.

Note: For an undirected graph, the matrix $L$ is symmetric, meaning the eigenvalues will all be real.

## Calculation of Eigenvectors
For a square matrix $A$, eigenvalues, $\lambda$, and eigenvectors, $x$, are definded by the scalar and vector solutions to $Ax = \lambda x$. Meaning that for any point on an eigenvector, a linear operator (represented by multiplication with the matrix $A$) will only scale this point. For problems involving differentiable equations, this usually represents some form of equality. With consideration of the Laplacian matrix (which represents all one-step possible paths), the eigenvectors summarize the importance of a node in these random walks since there are multiple eigenvectors for a unique eigenvalue and eigenvalues are used to declare two matrices similar. Thus, nodes are clustered based on this similarity characteristic. 

Eigenvalues and Eigenvectors are both calculated in one process using the QR algorithm. This algorithm iteratively finds the QR decomposition of the matrix, reverses the order of $Q$ and $R$, and converges to an upper-triangular matrix $R'$ and an orthogonal matrix $Q'$ where the diagonal of $R'$ is the eigenvalues of $A$ and the columns of $Q'$, which is the product of all $Q_k$, are the eigenvectors of $A$.
\begin{align*}
A &= Q_1 * R_1 \\
A_{k+1} &= R_k * Q_k \\
A_{k+1} &= Q_{k+1} * R_{k+1} \\
\end{align*}
Each $A_k$ is similar to the previous (meaning they have the same eigenvalues), but the steps converge (slowly) to an upper-triangular matrix.

For QR decomposition, the Gram-Schmidt process was implemented to find an orthogonal basis of $A$, $Q$, and then $A$ = $QR$ means $Q^{-1} * A = R$; thankfully since $Q$ is orthogonal, its inverse is its transpose. The Gram-Schmidt process for a matrix $A$ is defined by finding the normal vector that represents the $i$th column of $A$ and subtracting this vector's projection onto the previous orthonormal vectors. Therefore the new vector is orthonormal with respect to the other vectors. Faster, more numerically stable (less propagtion of error) methods exist for calcualting an orthogonal basis of a matrix, but this is an easily implemented method. 

To speed up the convergence of the matrix, we can convert the matrix $A$ into a similar matrix $H$ in Hessenberg form; meaning every index below the subdiagonal (diagonal below the main diagonal) is zero. This is done through Householder reductions (https://math.la.asu.edu/~gardner/QR.pdf). Householder matrices reflect a matrix across a vector $u$. For a vector $u$, the Householder matrix is $R = I - \frac{2uuT}{||u||^2}$. Set the $k$th column in $A$ to $v$. We want to find a matrix so that the $PAP^-1$ reduces the elements of column $k$ below the subdiagonal to 0. This will be a vector $v'$ such that $Hv' = [||v|| 0 0 0 \cdots]$; in particular $v'$ = $v - [||v|| 0 0 0 \cdots]$. Using $v'$ we can find $R$ the Housholder matrix that will reflect our matrix so that the column in question is zeroed - the matrix P is the identity matrix up to column $k$ and then $R$ thereafter. This maintains the zeros we already have. So $H := PHP^T$ and $Q := QP$. After iterating through all columns, $H$ is in Hessenberg form and $Q$ is a orthogonal matrix that contains all of these transformations. 

The last modification that helps the convergence of the QR-algorithm is shifts of the matrix. There are proofs of this and faster ways of going about its implementation, but prior to the QR decomposition you subtract by $cI$ where $c$ is an approximation of an eigenvalue; then after the RQ multiplication, we add $cI$. The last entry of the diagonal is often used for this $c$, in this implemenation, we iterate through the last three diagonals. 

Finally, the convergence of the QR-algorithm is not entirely upper-triangular if there are complex eigenvalues. If this occurs, then there will be a 2x2 matrix on the diagonal of the form:
$$
\left(\begin{array}{cc} 
Re(a) & Im(b)\\
Im(a) & Re(b)
\end{array}\right)
$$ 
In order to detect these and help convergence, we look for the singular Im(a) below the diagonal and replace this 2x2 matrix with its eigenvalues calculated directly using the characteristic equation and quadratic formula.

## K-Means
K-Means clustering of a set of vectors is done by randomly assigning the vectors to $k$ clusters. The means of each cluster is used to define their centroid. Each observation (vector) is then reassigned to the cluster whose centroid it is closest to. This is done repeatedly until a maximum number of iterations is reached or the centroids have moved less than some tolerance amount (1e-6 used here). 

## Data
The data used for this clustering is 494-Bus in a Matrix Market file while stores the sparse matrix as a list of $i$ $j$ value. Since we know that this matrix will be symmetric, we will have to place the value at both $i, j$ and $j, i$. 






























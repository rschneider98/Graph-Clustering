#include "pch.h"
#include <vector>
#include "../Graph-Clustering/LinAlg.h"

using namespace LinAlg;


// Matrix Assignment Operations and Exceptions
TEST(MatrixAssignment, SquareMatrix) {
	std::vector<std::vector<double>> myVect{
		{1, 2, 3},
		{0, 5, 6},
		{0, 0, 9}
	};

	Matrix myMatrix(3, 3);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			myMatrix[i][j] = myVect[i][j];
		}
	}

	EXPECT_EQ(Matrix(myVect), myMatrix);
}

TEST(MatrixAssignment, LargerMatrix) {
	std::vector<std::vector<double>> myVect{
		{0.480646, 0.723372, 0.593985, 0.981938, 0.609705, 0.303921, 0.554738, 0.606491, 0.782623, 0.500412, 0.319764, 0.488999, 0.238063, 0.568017, 0.202253, 0.738818, 0.65568, 0.561607, 0.475934, 0.230066},
		{0.637454, 0.382096, 0.623483, 0.091715, 0.599125, 0.551827, 0.696237, 0.426812, 0.96492, 0.633536, 0.417538, 0.981714, 0.136535, 0.0753173, 0.254207, 0.454222, 0.369227, 0.228164, 0.303688, 0.763199}, 
		{0.523268, 0.12527, 0.211621, 0.763833, 0.459339, 0.381219, 0.175082, 0.421672, 0.186365, 0.158586, 0.910018, 0.953743, 0.948085, 0.555681, 0.210836, 0.555707, 0.438315, 0.0685704, 0.871863, 0.766507},
		{0.376932, 0.624592, 0.787269, 0.779324, 0.815139, 0.818786, 0.166275, 0.0821495, 0.972884, 0.543581, 0.0749341, 0.0930281, 0.855828, 0.0857746, 0.559331, 0.72236, 0.481328, 0.263146, 0.713337, 0.907432},
		{0.314901, 0.834525, 0.0295366, 0.255764, 0.872284, 0.140806, 0.702131, 0.923245, 0.661852, 0.78355, 0.589918, 0.767509, 0.103397, 0.843028, 0.250093, 0.982382, 0.605785, 0.347606, 0.908427, 0.928431},
		{0.267003, 0.00963683, 0.242592, 0.510597, 0.387309, 0.985655, 0.950464, 0.347942, 0.461432, 0.686107, 0.733452, 0.90892, 0.837378, 0.53858, 0.813237, 0.394963, 0.414237, 0.231188, 0.245215, 0.788132}, 
		{0.508328, 0.846712, 0.635553, 0.724944, 0.234765, 0.534687, 0.899005, 0.658451, 0.461723, 0.475042, 0.903114, 0.251117, 0.367136, 0.451206, 0.931704, 0.479769, 0.558708, 0.816832, 0.00822299, 0.559773}, 
		{0.121444, 0.998391, 0.0439992, 0.511351, 0.310018, 0.427793, 0.2962, 0.284747, 0.813222, 0.977871, 0.334798, 0.149385, 0.506296, 0.676057, 0.148192, 0.430538, 0.447907, 0.637175, 0.00299187, 0.371268}, 
		{0.835097, 0.906152, 0.311052, 0.964822, 0.99753, 0.230838, 0.588, 0.234235, 0.523459, 0.222493, 0.999718, 0.216285, 0.27798, 0.384954, 0.894824, 0.365934, 0.752279, 0.336348, 0.54983, 0.202736}, 
		{0.391516, 0.689542, 0.0964235, 0.82729, 0.816098, 0.349749, 0.136713, 0.0034692, 0.235773, 0.612756, 0.826206, 0.0783768, 0.318581, 0.982396, 0.590085, 0.683494, 0.806147, 0.243157, 0.600364, 0.38258},
		{0.900539, 0.546104, 0.612682, 0.142531, 0.532284, 0.230009, 0.591491, 0.11314, 0.647808, 0.214789, 0.678971, 0.349562, 0.504082, 0.690763, 0.518021, 0.788729, 0.0322757, 0.998492, 0.276714, 0.265212},
		{0.750598, 0.548224, 0.183729, 0.45765, 0.302414, 0.22267, 0.539396, 0.315031, 0.318276, 0.026276, 0.595553, 0.835659, 0.850572, 0.479208, 0.661657, 0.642833, 0.392885, 0.604177, 0.076941, 0.0579934}, 
		{0.265057, 0.987728, 0.813034, 0.638616, 0.753556, 0.805979, 0.158731, 0.873247, 0.160865, 0.936093, 0.264437, 0.91975, 0.346135, 0.65523, 0.712024, 0.0729487, 0.263517, 0.133598, 0.290597, 0.674474},
		{0.509967, 0.612865, 0.0985418, 0.570723, 0.330374, 0.628735, 0.790442, 0.477042, 0.561413, 0.12048, 0.283075, 0.869977, 0.919701, 0.558329, 0.358551, 0.576045, 0.0663167, 0.771675, 0.61001, 0.0763916}, 
		{0.838088, 0.59681, 0.548956, 0.731242, 0.527769, 0.275556, 0.831319, 0.0961716, 0.438063, 0.111111, 0.399226, 0.0562126, 0.179879, 0.188807, 0.201835, 0.141319, 0.250288, 0.0267932, 0.766619, 0.545167}, 
		{0.633052, 0.0791792, 0.241671, 0.808902, 0.458657, 0.528578, 0.774682, 0.226245, 0.829927, 0.58301, 0.239444, 0.338437, 0.567354, 0.0296385, 0.33278, 0.165697, 0.203432, 0.712512, 0.350716, 0.659964}, 
		{0.254906, 0.103733, 0.385037, 0.632591, 0.674063, 0.968198, 0.245707, 0.313566, 0.577587, 0.86766, 0.588749, 0.609057, 0.978924, 0.377698, 0.38224, 0.814136, 0.577168, 0.776772, 0.762711, 0.322188}, 
		{0.191788, 0.829244, 0.715179, 0.462238, 0.971211, 0.0258569, 0.665818, 0.926993, 0.299897, 0.0649925, 0.856506, 0.0299418, 0.553075, 0.10885, 0.367502, 0.637471, 0.288002, 0.75563, 0.866189, 0.379059}, 
		{0.988907, 0.704244, 0.539547, 0.811168, 0.46668, 0.550818, 0.0895863, 0.0807472, 0.274611, 0.439863, 0.419183, 0.820577, 0.874013, 0.246557, 0.924564, 0.0804644, 0.345608, 0.567286, 0.892554, 0.761124}, 
		{0.73947, 0.898173, 0.755457, 0.589409, 0.343765, 0.722955, 0.0379529, 0.253949, 0.750728, 0.396843, 0.570026, 0.455341, 0.662669, 0.980269, 0.0885867, 0.226841, 0.283358, 0.898229, 0.180749, 0.84077}
	};

	Matrix myMatrix(20, 20);
	for (int i = 0; i < 20; i++) {
		for (int j = 0; j < 20; j++) {
			myMatrix[i][j] = myVect[i][j];
		}
	}

	EXPECT_EQ(Matrix(myVect), myMatrix);
}

TEST(MatrixAssignment, WrongSizeVector) {
	std::vector<std::vector<double>> myVect{
		{1, 2, 3},
		{0, 5},
		{0, 0, 9}
	};

	EXPECT_ANY_THROW(Matrix m(myVect));
}

TEST(MatrixAssignment, NegativeRows) {
	EXPECT_ANY_THROW(Matrix m(-1, 4));
}

TEST(MatrixAssignment, NegativeCols) {
	EXPECT_ANY_THROW(Matrix m(3, -1));
}

// Correct Evaluation of Matrix Operations
TEST(MatrixOps, TransposeSquare) {
	std::vector<std::vector<double>> myVect1{
		{1, 2, 3},
		{0, 5, 6},
		{0, 0, 9}
	};
	std::vector<std::vector<double>> myVect2{
		{1, 0, 0},
		{2, 5, 0},
		{3, 6, 9}
	};
	EXPECT_EQ(Matrix(myVect1).T(), Matrix(myVect2));
}

TEST(MatrixOps, TransposeNonSquare) {
	std::vector<std::vector<double>> myVect1{
		{1, 2, 3, 2},
		{0, 5, 6, 4},
		{0, 0, 9, 7}
	};
	std::vector<std::vector<double>> myVect2{
		{1, 0, 0},
		{2, 5, 0},
		{3, 6, 9},
		{2, 4, 7}
	};
	EXPECT_EQ(Matrix(myVect1).T(), Matrix(myVect2));
}

TEST(MatrixOps, Determinant2) {
	std::vector<std::vector<double>> myVect{
		{1, 2},
		{3, 5}
	};
	double det = (1 * 5) - (2 * 3);
	EXPECT_EQ(Matrix(myVect).Det(), det);
}

TEST(MatrixOps, Determinant3) {
	std::vector<std::vector<double>> myVect{
		{1, 2, 3},
		{0, 5, 6},
		{0, 0, 9}
	};
	double det = 1 * 5 * 9;
	EXPECT_EQ(Matrix(myVect).Det(), det);
}

TEST(MatrixOps, Determinant4) {
	std::vector<std::vector<double>> myVect{
		{0.83462, 0.699265, 0.930962, 0.863479},
		{0.473505, 0.691988, 0.747999, 0.225765},
		{0.933097, 0.105651, 0.280316, 0.0671968},
		{0.496422, 0.556214, 0.727784, 0.789878}
	};
	double det = 0.00032226;
	EXPECT_NEAR(Matrix(myVect).Det(), det, 1e-6);
}

TEST(MatrixOps, Addition1) {
	std::vector<std::vector<double>> myVect1{
		{1, 2, 3},
		{0, 5, 6},
		{0, 0, 9}
	};

	std::vector<std::vector<double>> myVect2{
		{1, 0, 0},
		{2, 5, 0},
		{3, 6, 9}
	};

	std::vector<std::vector<double>> myVect3{
		{2, 2, 3},
		{2, 10, 6},
		{3, 6, 18}
	};

	Matrix res = Matrix(myVect1) + Matrix(myVect2);
	Matrix exp_res(myVect3);
	EXPECT_EQ(res, exp_res);
}

TEST(MatrixOps, Subtraction1) {
	std::vector<std::vector<double>> myVect1{
		{1, 2, 3},
		{0, 5, 6},
		{0, 0, 9}
	};

	std::vector<std::vector<double>> myVect2{
		{1, 2, 3},
		{0, 5, 6},
		{0, 0, 9}
	};

	std::vector<std::vector<double>> myVect3{
		{0, 0, 0},
		{0, 0, 0},
		{0, 0, 0}
	};

	Matrix res = Matrix(myVect1) - Matrix(myVect2);
	Matrix exp_res(myVect3);
	EXPECT_EQ(res, exp_res);
}

TEST(MatrixOps, Multiplication1) {
	std::vector<std::vector<double>> myVect1{
		{0.73054, 0.715638, 0.673198, 0.0587931},
		{0.189321, 0.447365, 0.596879, 0.448019},
		{0.908993, 0.847864, 0.237653, 0.708224}
	};

	std::vector<std::vector<double>> myVect2{
		{0.751884}, 
		{0.039623},
		{0.811679},
		{0.919456}
	};

	std::vector<std::vector<double>> myVect3{
		{1.178115}, 
		{1.056481},
		{1.561131}
	};

	Matrix res = Matrix(myVect1) * Matrix(myVect2);
	Matrix exp_res(myVect3);
	EXPECT_EQ(res, exp_res);
}

TEST(MatrixOps, RREFBasicSquare) {
	std::vector<std::vector<double>> myVect1{
		{1, 2, 3},
		{0, 5, 6},
		{0, 0, 9}
	};
	EXPECT_EQ(Matrix(myVect1).RREF(), Eye(3));
}

TEST(MatrixOps, RREFLargerSquare) {
	std::vector<std::vector<double>> myVect1{
		{0.0510578, 0.398722, 0.736084, 0.441141, 0.604087, 0.0981771}, 
		{0.729175, 0.809315, 0.582718, 0.968245, 0.516376, 0.619702},
		{0.525811, 0.639631, 0.33082, 0.535713, 0.58745, 0.270752},
		{0.256436, 0.277358, 0.951789, 0.0650985, 0.00484161, 0.0525908},
		{0.513702, 0.522293, 0.544872, 0.410142, 0.325382, 0.646848},
		{0.143156, 0.309489, 0.970112, 0.326438, 0.224972, 0.885485}
	};

	EXPECT_EQ(Matrix(myVect1).RREF(), Eye(6));
}

TEST(MatrixOps, RREFNonSquare1) {
	std::vector<std::vector<double>> myVect1{
		{0.73054, 0.715638, 0.673198, 0.0587931},
		{0.189321, 0.447365, 0.596879, 0.448019},
		{0.908993, 0.847864, 0.237653, 0.708224}
	};

	std::vector<std::vector<double>> myVect2{
		{1, 0, 0, -2.412516},
		{0, 1, 0, 3.793921},
		{0, 0, 1, -1.327754}
	};

	EXPECT_EQ(Matrix(myVect1).RREF(), Matrix(myVect2));
}

TEST(MatrixOps, RREFNonSquare2) {
	std::vector<std::vector<double>> myVect1{
		{1, -5, 8},
		{1, -2, 1}, 
		{2, -1, -5},
		{2, 4, 7}
	};

	std::vector<std::vector<double>> myVect2{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{0, 0, 0}
	};

	EXPECT_EQ(Matrix(myVect1).RREF(), Matrix(myVect2));
}

TEST(MatrixOps, RREFLargerNonSquare) {
	std::vector<std::vector<double>> myVect1{
		{1, -5, 8, 3}, 
		{1, -2, 1, 2},
		{2, -1, -5, 6}
	};

	std::vector<std::vector<double>> myVect2{
		{1, 0, -3.666667, 0},
		{0, 1, -2.333333, 0}, 
		{0, 0, 0, 1}
	};

	EXPECT_EQ(Matrix(myVect1).RREF(), Matrix(myVect2));
}

TEST(MatrixOps, SolveID) {
	std::vector<double> myVect1{1, 2, 3};
	Vector v(myVect1);
	EXPECT_EQ(Eye(3).Solve(v), v);
}

TEST(MatrixOps, SolveBasic) {
	std::vector<double> myVect1{1, 2, 3};
	std::vector<std::vector<double>> myVect2{
		{1, 2, 3},
		{5, 7, 9},
		{11, 13, 17}
	};
	std::vector<double> myVect3{-0.5, 0, 0.5};
	Vector v(myVect1);
	Matrix test1(myVect2);
	EXPECT_EQ(test1.Solve(v), Vector(myVect3));
}

TEST(MatrixOps, GramSchmidt1) {
	std::vector<std::vector<double>> myVect1{
		{3, 2},
		{1, 2}
	};
	std::vector<std::vector<double>> myVect2{
		{0.948683, -0.316228},
		{0.316228, 0.948683}
	};
	EXPECT_EQ(Matrix(myVect1).GramSchmidt(), Matrix(myVect2));
}

TEST(MatrixOps, GramSchmidt2) {
	std::vector<std::vector<double>> myVect1{
		{0.749935, 0.0239392, 0.909929},
		{0.233785, 0.0521392, 0.192435},
		{0.632099, 0.611084, 0.748108}
	};
	std::vector<std::vector<double>> myVect2{
		{0.743784, -0.623818, 0.240075}, 
		{0.231868, -0.096072, -0.967992}, 
		{0.626915, 0.775643, 0.073186}
	};
	EXPECT_EQ(Matrix(myVect1).GramSchmidt(), Matrix(myVect2));
}

TEST(MatrixOps, isTrigFalse) {
	std::vector<std::vector<double>> myVect1{
		{0.749935, 0.0239392, 0.909929},
		{0.233785, 0.0521392, 0.192435},
		{0.632099, 0.611084, 0.748108}
	};
	EXPECT_EQ(Matrix(myVect1).isTrig(), false);
}

TEST(MatrixOps, isTrigTrue) {
	std::vector<std::vector<double>> myVect1{
		{0.749935, 0.0239392, 0.909929},
		{0,        0.0521392, 0.192435},
		{0,        0,         0.748108}
	};
	EXPECT_EQ(Matrix(myVect1).isTrig(), true);
}

TEST(MatrixOps, isTrigNonSquareTrue) {
	std::vector<std::vector<double>> myVect1{
		{1, 2, 3, 5},
		{0, 5, 7, 11},
		{0, 0, 11, 13}
	};
	EXPECT_EQ(Matrix(myVect1).isTrig(), true);
}

TEST(MatrixOps, isTrigRREF1) {
	std::vector<std::vector<double>> myVect1{
		{1, 2, 3, 5},
		{3, 5, 7, 11},
		{5, 7, 11, 13}
	};
	EXPECT_EQ(Matrix(myVect1).RREF().isTrig(), true);
}

TEST(MatrixOps, isTrigRREF2) {
	std::vector<std::vector<double>> myVect1{
		{0.135451, 0.923285, 0.260706, 0.358978, 0.0277942},
		{0.574996, 0.922701, 0.478299, 0.854408, 0.77014},
		{0.416416, 0.723604, 0.971133, 0.960431, 0.371329},
		{0.770157, 0.894326, 0.743106, 0.207068, 0.505726},
		{0.623625, 0.580306, 0.865516, 0.939035, 0.916691}
	};
	EXPECT_EQ(Matrix(myVect1).RREF().isTrig(), true);
}
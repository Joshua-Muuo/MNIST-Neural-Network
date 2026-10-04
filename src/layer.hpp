#pragma once
#include "types.hpp"
#include <Eigen/Dense>

using Matrix = Eigen::MatrixXd; 
class Layer
{
	virtual Matrix forward(const Matrix& inputs);

	virtual Matrix backward(const Matrix& inputs, f32 learningRate);
};

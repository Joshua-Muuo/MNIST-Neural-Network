#pragma once
#include "types.hpp"
#include <Eigen/Dense>

using Matrix = Eigen::MatrixXf; 
class Layer
{
public:
	virtual Matrix forward(const Matrix& inputs) = 0;
	virtual Matrix backward(const Matrix& inputs, f32 learningRate) = 0;

	virtual ~Layer() = default;
};

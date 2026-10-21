#include "softmax.hpp"
#include <iostream>
#include <print>

SoftMax::SoftMax(u32 inputSize, u32 outputSize)
{
	
}

Matrix SoftMax::forward(const Matrix& inputs)
{
	f32 sum{};
	Eigen::VectorXf maxValues{inputs.rowwise().maxCoeff()};

	Matrix inputCopy{(inputs.colwise() - maxValues).array().exp()};

	Eigen::VectorXf rowSums{inputCopy.rowwise().sum()};
	inputCopy	 = inputCopy.array().colwise() / rowSums.array();

	return inputCopy;
}

Matrix SoftMax::backward(const Matrix& dOutputs) 
{	
	return dOutputs;
}

void SoftMax::applyGradientDescent(f32 learningRate)
{
	return;
}

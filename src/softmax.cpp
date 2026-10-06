#include "softmax.hpp"

SoftMax::SoftMax(u32 inputSize, u32 outputSize)
{
	
}

Matrix SoftMax::forward(const Matrix& inputs)
{
	Matrix inputCopy{inputs};

	f32 sum{};
	for (auto& value : inputCopy.reshaped())
	{
		value = std::exp(value);
		sum += value;
	}

	for (auto& value : inputCopy.reshaped())
	{
		value /= sum;
	}

	return inputCopy;
}
Matrix SoftMax::backward(const Matrix& inputs, f32 learningRate) 
{
	return inputs;	
}

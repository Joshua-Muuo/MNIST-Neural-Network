#include "relu_layer.hpp"

ReLULayer::ReLULayer(u32 inputSize, u32 outputSize)
{
	
}

Matrix ReLULayer::forward(const Matrix& inputs)
{
	Matrix inputCopy{inputs};
	for (auto& value : inputCopy.reshaped())
	{
		if (value < 0.0f) value = 0.0f;
	}

	return inputCopy;
}
Matrix ReLULayer::backward(const Matrix& inputs, f32 learningRate) 
{
	return inputs;	
}

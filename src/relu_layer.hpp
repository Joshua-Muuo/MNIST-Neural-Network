#pragma once
#include "layer.hpp"

class ReLULayer : public Layer
{
public:
	ReLULayer(u32 inputSize, u32 outputSize);

	Matrix forward(const Matrix& inputs) override;
	Matrix backward(const Matrix& inputs, f32 learningRate) override;
};

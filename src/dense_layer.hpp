#pragma once
#include "layer.hpp"

class DenseLayer : public Layer
{
public:
	DenseLayer(u32 inputSize, u32 outputSize);

	Matrix forward(const Matrix& inputs) override;
	Matrix backward(const Matrix& inputs, f32 learningRate) override;
};

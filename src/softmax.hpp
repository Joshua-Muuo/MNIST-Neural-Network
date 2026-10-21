#pragma once
#include "layer.hpp"

class SoftMax : public Layer
{
public:
	SoftMax(u32 inputSize, u32 outputSize);

	Matrix forward(const Matrix& inputs) override;
	Matrix backward(const Matrix& dOutputs) override;
	void applyGradientDescent(f32 learningRate) override;

private:
	Matrix mOutput, mDOutputs;
};


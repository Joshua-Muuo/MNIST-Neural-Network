#include "Core/relu_layer.hpp"

ReLULayer::ReLULayer(u32 inputSize, u32 outputSize)
{
	
}

Matrix ReLULayer::forward(const Matrix& inputs)
{
	mAPrev = inputs;
	return inputs.unaryExpr([](f32 x) { return x >= 0.0f ? x : 0.1f * x; } );
}

Matrix ReLULayer::backward(const Matrix& inputs) 
{
	return inputs.binaryExpr(mAPrev, 
			[](f32 dy, f32 x) 
			{ return x > 0.0f ? dy : 0.0f; 
			});	
}

void ReLULayer::applyGradientDescent(f32 learningRate)
{
	return;
}


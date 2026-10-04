#include "dense_layer.hpp"

DenseLayer::DenseLayer(u32 inputSize, u32 outputSize):
mWeights{outputSize, inputSize}, mBiases{1, outputSize}
{
	mWeights.setRandom();
	mBiases.setZero();
}

Matrix DenseLayer::forward(const Matrix& inputs)
{
	return inputs * mWeights.transpose() + mBiases;
}

Matrix DenseLayer::backward(const Matrix& inputs, f32 learningRate)
{
	return inputs;
}

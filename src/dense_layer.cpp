#include "dense_layer.hpp"
#include "utils.hpp"
#include <functional>
#include <print>

DenseLayer::DenseLayer(u32 inputSize, u32 outputSize):
mWeights{Matrix::NullaryExpr(outputSize, inputSize, std::ref(Utils::randFloat))}, 
mBiases{outputSize}
{
	// mWeights.setRandom();
	// for (auto& x : mWeights.reshaped())
		// x = Utils::randFloat();
	mBiases.setZero();
}

Matrix DenseLayer::forward(const Matrix& inputs)
{
	auto c{(i32)(inputs.cols())};
	std::println("inputs col is {}", c);
	std::println("weights transpose rows is {}", (u32)mWeights.cols());

	Matrix product{inputs * mWeights.transpose()};
	product.rowwise() += mBiases;

	return product;
}

Matrix DenseLayer::backward(const Matrix& inputs, f32 learningRate)
{
	return inputs;
}

#include "dense_layer.hpp"
#include "utils.hpp"
#include <functional>
#include <print>

DenseLayer::DenseLayer(u32 inputSize, u32 outputSize):
mWeights{Matrix::NullaryExpr(outputSize, inputSize, [](){return Utils::randFloat() * 2 - 1; })}, 
mBiases{outputSize}
{
	// mWeights.setRandom();
	// for (auto& x : mWeights.reshaped())
		// x = Utils::randFloat();
	mBiases.setZero();
}

Matrix DenseLayer::forward(const Matrix& inputs)
{

	mInputs = inputs;
	auto c{(i32)(inputs.cols())};
	std::println("inputs col is {}", c);
	std::println("weights transpose rows is {}", (u32)mWeights.cols());

	Matrix product{inputs * mWeights.transpose()};
	product.rowwise() += mBiases;

	return product;
}

Matrix DenseLayer::backward(const Matrix& dOutputs)
{    
	const f32 batchSize{static_cast<f32>(dOutputs.rows())};
	mDWeights.noalias() = (mInputs.transpose() * dOutputs).transpose() / batchSize; 
	mDBiases.noalias()  = dOutputs.colwise().sum() / batchSize;

	return dOutputs * mWeights;
}

void DenseLayer::applyGradientDescent(f32 learningRate)
{
	mWeights -= learningRate * mDWeights;
	mBiases  -= learningRate * mDBiases;
}


const Matrix& DenseLayer::getWeights() const
{
	return mWeights;
}

const Eigen::RowVectorXf& DenseLayer::getBiases() const
{
	return mBiases;
}

void DenseLayer::setWeights(const Matrix& mat)
{
	// assert(mWeights.rows() == mat.rows() && mWeights.cols() == mat.cols());
	mWeights = mat;
}

void DenseLayer::setBiases(const Eigen::RowVectorXf& rowVector)
{
	// assert(mBiases.cols() == rowVector.cols());
	mBiases = rowVector;
}

void DenseLayer::saveData()
{
	Utils::saveWeights(mWeights);
	Utils::saveBiases(mBiases);
}


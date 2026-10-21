#pragma once
#include "layer.hpp"
#include <expected>

class DenseLayer : public Layer
{
public:
	DenseLayer(u32 inputSize, u32 outputSize);

	Matrix forward(const Matrix& inputs) override;
	Matrix backward(const Matrix& dOutputs) override;
	void applyGradientDescent(f32 learningRate) override;
	void saveData() override;


	const Matrix& getWeights() const;
	const Eigen::RowVectorXf& getBiases() const;
	void setWeights(const Matrix& mat);
	void setBiases(const Eigen::RowVectorXf& row);


private:
	Matrix mWeights, mDWeights, mInputs;
	Eigen::RowVectorXf mBiases, mDBiases;
};

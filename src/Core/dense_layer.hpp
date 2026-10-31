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

	u32 inputSize() const { return mWeights.cols(); }
	u32 outputSize() const { return mWeights.rows(); }
	
	u32 rows() const { return mWeights.rows(); }
	u32 cols() const { return mWeights.cols(); }


private:
	Matrix mWeights, mDWeights, mAPrev;
	Eigen::RowVectorXf mBiases, mDBiases;
};

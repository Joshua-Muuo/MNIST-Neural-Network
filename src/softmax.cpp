#include "softmax.hpp"

SoftMax::SoftMax(u32 inputSize, u32 outputSize)
{
	
}

Matrix SoftMax::forward(const Matrix& inputs)
{
	Matrix inputCopy{inputs};

	f32 sum{};
	Eigen::VectorXf maxValues{inputCopy.rowwise().maxCoeff()};
	for (Eigen::Index i{}; i < inputCopy.rows(); ++i)
	{
		auto row{inputCopy.row(i)};
		f32 maxRowValue{maxValues[i]};
		for (f32& value : row)
		{
			value -= maxRowValue;
			value = std::exp(value);
		}
	}

	for (Eigen::Index i{}; i < inputCopy.rows(); ++i)
	{
		auto row{inputCopy.row(i)};
		f32 rowSum{row.sum()};

		for (f32& value : row)
		{
			value /= rowSum;
		}
	}

	return inputCopy;
}
Matrix SoftMax::backward(const Matrix& inputs, f32 learningRate) 
{
	return inputs;	
}

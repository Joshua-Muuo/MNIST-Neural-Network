#pragma once
#include "layer.hpp"
#include <initializer_list>
#include <memory>
#include <vector>

class NeuralNetwork
{
public:
	NeuralNetwork(std::initializer_list<u32> layerSizes);

	void train(u32 epochs);
	f32 testAccuracy();
	void loadWeightsAndBiases(const std::string& pathToModel);

	u32 makePrediction(const Matrix& mat) const;
	Eigen::VectorXi makePredictions(const Matrix& mat) const;
private:
	Matrix forward(Matrix input) const;
	Matrix forwardWithoutSoftmax(Matrix input) const;
	void backward(Matrix predOutputs, std::span<f32> labels) const;
	void applyGradientDescent();

	void saveData() const;

	std::vector<std::unique_ptr<Layer>> mLayers;
	const u32 BATCH_SIZE{128u};
};

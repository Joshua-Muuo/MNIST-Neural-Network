#pragma once
#include "layer.hpp"
#include <initializer_list>
#include <memory>
#include <vector>

class NeuralNetwork
{
public:
	NeuralNetwork(std::initializer_list<u32> layerSizes);

	void train();
private:
	Matrix forward(Matrix input);
	void backward(Matrix predOutputs, std::span<f32> labels);
	void applyGradientDescent();

	void saveData();

	std::vector<std::unique_ptr<Layer>> mLayers;
	const u32 BATCH_SIZE{128u};
};

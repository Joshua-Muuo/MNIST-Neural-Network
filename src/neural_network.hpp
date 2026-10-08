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
	void backward(Matrix batchInputs, Matrix predOutputs, std::span<u8> labels);

	std::vector<std::unique_ptr<Layer>> mLayers;
	const u32 BATCH_SIZE{100u};
};

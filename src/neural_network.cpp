#pragma once
#include "neural_network.hpp"
#include "dense_layer.hpp"
#include "relu_layer.hpp"

NeuralNetwork::NeuralNetwork(std::initializer_list<u32> layerSizes)
{
	for (usize i{1}; i < layerSizes.size(); ++i)
	{
		u32 inputSize{layerSizes.begin()[i - 1]}, outputSize{layerSizes.begin()[i]};
		mLayers.emplace_back(std::make_unique<DenseLayer>(inputSize, outputSize));
		mLayers.emplace_back(std::make_unique<ReLULayer>(inputSize, outputSize));
	}
}

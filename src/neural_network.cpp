#include "neural_network.hpp"
#include "dense_layer.hpp"
#include "relu_layer.hpp"
#include <iostream>

NeuralNetwork::NeuralNetwork(std::initializer_list<u32> layerSizes)
{
	for (usize i{1}; i < layerSizes.size(); ++i)
	{
		u32 inputSize{layerSizes.begin()[i - 1]}, outputSize{layerSizes.begin()[i]};
		mLayers.emplace_back(std::make_unique<DenseLayer>(inputSize, outputSize));
		mLayers.emplace_back(std::make_unique<ReLULayer>(inputSize, outputSize));
	}
}

void NeuralNetwork::forward()
{
	Matrix input{1, 784};
	input.setRandom();
	for (auto& ptr : mLayers)
	{
		input = ptr->forward(input);
	}

	std::cout << "input is: \n" << input << "\n\n\n";
}

void NeuralNetwork::backward()
{
}

void NeuralNetwork::train()
{
	forward();
}

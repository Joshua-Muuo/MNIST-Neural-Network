#include "neural_network.hpp"
#include "dense_layer.hpp"
#include "relu_layer.hpp"
#include "softmax.hpp"
#include <iostream>
#include <memory>
#include <print>
#include "utils.hpp"

NeuralNetwork::NeuralNetwork(std::initializer_list<u32> layerSizes)
{
	for (usize i{1}; i < layerSizes.size(); ++i)
	{
		u32 inputSize{layerSizes.begin()[i - 1]}, outputSize{layerSizes.begin()[i]};
		mLayers.emplace_back(std::make_unique<DenseLayer>(inputSize, outputSize));
		mLayers.emplace_back(std::make_unique<ReLULayer>(inputSize, outputSize));
	}

	mLayers.emplace_back(std::make_unique<SoftMax>(1, 10));
}

Matrix NeuralNetwork::forward(Matrix input)
{
	// input.setRandom();
	for (auto& ptr : mLayers)
	{
		input = ptr->forward(input);
	}

	return input;
}

void NeuralNetwork::backward(Matrix batchInputs, Matrix predOutputs, std::span<u8> labels)
{
}

void NeuralNetwork::train()
{
	auto imageBuffer = Utils::loadImage("data/train-images-idx3-ubyte.gz");
	auto labelBuffer = Utils::loadLabel("data/train-labels-idx1-ubyte.gz");

	// header is 16 bytes and is stored in big-endian
	
	u32* num;
	for (usize i{}; i < 16; i += sizeof(u32))
	{
		num = std::start_lifetime_as<u32>(&imageBuffer[i]);
		*num = std::byteswap(*num);
	}

	num = (u32*)&imageBuffer[4];

	const char* BLACK{"\033[40m"}, *WHITE{"\033[47m"}, *RESET{"\033[0m"};
	for (usize i{784 * 9}; i < 784 * 10; ++i)
	{
		if ((i) % 28 == 0 && i != 0)
			std::println("{}", RESET);
		// std::print("{}", (i - 16) % 28);
		std::print("\033[48;2;{};{};{}m  {}", imageBuffer[i], imageBuffer[i], imageBuffer[i], RESET);
	}

	std::println("\n");
	// Matrix input{Matrix::NullaryExpr(100, 784, std::ref(Utils::randFloat))};

	Eigen::Map<Eigen::Matrix<u8, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> input{imageBuffer.data(), 100, 784};
	Matrix predOutput{forward(input)};



}

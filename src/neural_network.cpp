#include "neural_network.hpp"
#include "dense_layer.hpp"
#include "relu_layer.hpp"
#include "softmax.hpp"
#include <iostream>
#include <memory>
#include <print>
#include "utils.hpp"

using RowMajorMatrix = Eigen::Matrix<f32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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
	

	const char* BLACK{"\033[40m"}, *WHITE{"\033[47m"}, *RESET{"\033[0m"};
	
	u32 index{Utils::rand(imageBuffer.numberOfItems)};
	for (usize i{784 * index}; i < 784 * (index + 1); ++i)
	{
		if ((i) % 28 == 0 && i != 0)
			std::println("{}", RESET);
		// std::print("{}", (i - 16) % 28);
		std::print("\033[48;2;{};{};{}m  {}", imageBuffer.data[i], imageBuffer.data[i], imageBuffer.data[i], RESET);
	}

	std::println("\nNumber is {}", labelBuffer.data[index]);
	// Matrix input{Matrix::NullaryExpr(100, 784, std::ref(Utils::randFloat))};

	std::vector<f32> batch{Utils::loadBatch(imageBuffer.data)};
	Eigen::Map<RowMajorMatrix> input{batch.data(), BATCH_SIZE, 784};
	Matrix output{forward(input)};

	backward(input, output, {labelBuffer.data.data(), BATCH_SIZE});

	std::cout << "output is: \n\n " << output << "\n\n";

	
}

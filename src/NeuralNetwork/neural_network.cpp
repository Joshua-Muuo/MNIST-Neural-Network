#include "neural_network.hpp"
#include "dense_layer.hpp"
#include "relu_layer.hpp"
#include "softmax.hpp"
#include <iostream>
#include <iterator>
#include <memory>
#include <print>
#include "utils.hpp"

using RowMajorMatrixU = Eigen::Matrix<u32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

NeuralNetwork::NeuralNetwork(std::initializer_list<u32> layerSizes)
{
	u32 inputSize, outputSize;
	for (usize i{1}; i < layerSizes.size(); ++i)
	{
		inputSize  = layerSizes.begin()[i - 1];
		outputSize = layerSizes.begin()[i];

		mLayers.emplace_back(std::make_unique<DenseLayer>(inputSize, outputSize));
		mLayers.emplace_back(std::make_unique<ReLULayer>(inputSize, outputSize));
	}

	mLayers.emplace_back(std::make_unique<SoftMax>(outputSize, 10));
}

Matrix NeuralNetwork::forward(Matrix input)
{
	// input.setRandom();
	for (auto& ptr : mLayers)
		input = ptr->forward(input);

	return input;
}

void NeuralNetwork::backward(Matrix predOutputs, std::span<f32> labels)
{
	
	Eigen::Map<RowMajorMatrix> labelMatrixMap{labels.data(), predOutputs.rows(), 10};
	Matrix dOutput{predOutputs - labelMatrixMap};
	for (auto it{mLayers.rbegin()}; it != mLayers.rend(); ++it)
		dOutput = (*it)->backward(dOutput);
	
}

void NeuralNetwork::train(u32 epochs)
{
	auto imageBuffer = Utils::loadImage("data/train-images-idx3-ubyte.gz");
	auto labelBuffer = Utils::loadLabel("data/train-labels-idx1-ubyte.gz");

	// header is 16 bytes and is stored in big-endian
	

	
	u32 index{Utils::printImage(imageBuffer.data, {})};
	
	std::println("\nNumber is {}", labelBuffer.data[index]);
	// Matrix input{Matrix::NullaryExpr(100, 784, std::ref(Utils::randFloat))};


	for (u32 i{}; i < epochs * imageBuffer.numberOfItems; i += BATCH_SIZE)
	{
		std::println("i is {}", i);
		u32 batchSize{ BATCH_SIZE };
		if (imageBuffer.numberOfItems - i < BATCH_SIZE)
			batchSize = imageBuffer.numberOfItems - i;

		std::vector<f32> batch{Utils::loadBatchImages(imageBuffer.data, i, batchSize)};
		Eigen::Map<RowMajorMatrix> input{batch.data(), batchSize, 784};
		Matrix predOutput{forward(input)};

		std::vector<f32> labelBatch{Utils::loadBatchLabels(labelBuffer.data, i, batchSize)};
		backward(predOutput, labelBatch);
		applyGradientDescent();
	}
	saveData();

	// std::cout << "output is: \n\n " << output << "\n\n";

	
}

void NeuralNetwork::applyGradientDescent()
{
	const f32 learningRate{0.05f};
	for (auto& layer : mLayers)
		layer->applyGradientDescent(learningRate);
}

u8 NeuralNetwork::makePrediction(const Matrix& mat)
{
	Matrix predOutput{forward(mat)};
	f32 maxValue{-1.0f};
	usize maxIndex{};
	for (usize i{}; i < predOutput.cols(); ++i)
	{
		f32 value{predOutput(0, i)};
		if (value > maxValue) maxIndex = i;
	}

	return maxIndex;
}

void NeuralNetwork::saveData()
{
	for (const auto& layer : mLayers)
	{
		layer->saveData();
	}
}

void NeuralNetwork::loadWeightsAndBiases()
{
	DenseLayer* denseLayer1{dynamic_cast<DenseLayer*>(mLayers[0].get())};
	DenseLayer* denseLayer2{dynamic_cast<DenseLayer*>(mLayers[2].get())};

	denseLayer1->setWeights(Utils::loadWeights("layers/dense_layer_weights_100x784"));
	denseLayer1->setBiases(Utils::loadBiases("layers/dense_layer_biases_1x100"));

	denseLayer2->setWeights(Utils::loadWeights("layers/dense_layer_weights_10x100"));
	denseLayer2->setBiases(Utils::loadBiases("layers/dense_layer_biases_1x10"));

}

void NeuralNetwork::testAccuracy()
{
	loadWeightsAndBiases();

	auto imagesData = Utils::loadImage("data/t10k-images-idx3-ubyte.gz");
	auto labelsData = Utils::loadLabel("data/t10k-labels-idx1-ubyte.gz");

	std::vector<f32> imagesBuffer{Utils::loadBatchImages(imagesData.data, 0, imagesData.numberOfItems)};

	Eigen::Map<RowMajorMatrix> images{imagesBuffer.data(), imagesData.numberOfItems, 784};
	u32 randImageIndex{Utils::printImage(imagesBuffer)};


	u8 prediction{makePrediction(images.row(randImageIndex))};

	std::println("actual: {}", labelsData.data[randImageIndex]);
	std::println("prediction: {}", prediction);
	
}

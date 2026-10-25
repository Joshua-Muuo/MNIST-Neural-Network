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
		if (i < layerSizes.size() - 1)
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
	Utils::Data trainingImages{Utils::loadImage("data/train-images-idx3-ubyte.gz")};
	Utils::Data trainingLabels{Utils::loadLabel("data/train-labels-idx1-ubyte.gz")};

	// header is 16 bytes and is stored in big-endian
	

	
	u32 index{Utils::printImage(trainingImages.data, {})};
	
	std::println("\nNumber is {}", trainingLabels.data[index]);
	// Matrix input{Matrix::NullaryExpr(100, 784, std::ref(Utils::randFloat))};


	for (u32 epoch{}; epoch < epochs; ++epoch)
	{
		for (u32 i{}; i < trainingImages.numberOfItems; i += BATCH_SIZE)
		{
			u32 batchSize{ std::min(BATCH_SIZE, trainingImages.numberOfItems - i) };

			std::vector<f32> batch{Utils::loadBatchImages(trainingImages.data, i, batchSize)};
			Eigen::Map<RowMajorMatrix> input{batch.data(), batchSize, Utils::PIXELS_PER_IMAGE};
			Matrix predOutput{forward(input)};

			std::vector<f32> labelBatch{Utils::loadBatchLabels(trainingLabels.data, i, batchSize)};
			backward(predOutput, labelBatch);
			applyGradientDescent();
		}

		std::println("End of epoch {}", epoch + 1);
	}
	saveData();

	// std::cout << "output is: \n\n " << output << "\n\n";

	
}

void NeuralNetwork::applyGradientDescent()
{
	const f32 learningRate{0.01f};
	for (auto& layer : mLayers)
		layer->applyGradientDescent(learningRate);
}

u32 NeuralNetwork::makePrediction(const Matrix& mat)
{
	Matrix predOutput{forward(mat)};
	Eigen::Index predictedDigit{};

	predOutput.row(0).maxCoeff(&predictedDigit);
	return predictedDigit;
}

Eigen::VectorXi NeuralNetwork::makePredictions(const Matrix& mat)
{
	Matrix predOutputs{forward(mat)};
	Eigen::VectorXi maxIndices{mat.rows()};

	for (usize i{}; i < mat.rows(); ++i)
		mat.row(i).maxCoeff(&maxIndices[i]);
	
	return maxIndices;
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

	Utils::Data testingImages{Utils::loadImage("data/t10k-images-idx3-ubyte.gz")};
	Utils::Data testingLabels{Utils::loadLabel("data/t10k-labels-idx1-ubyte.gz")};

	std::vector<f32> imagesBuffer{Utils::loadBatchImages(testingImages.data, 0, testingImages.numberOfItems)};

	Eigen::Map<RowMajorMatrix> images{imagesBuffer.data(), testingImages.numberOfItems, 784};
	u32 randImageIndex{Utils::printImage(imagesBuffer)};
	std::println("prediction: {}", makePrediction(images.row(randImageIndex)));
	std::println("actual: {}\n\n", testingLabels.data[randImageIndex]);

	f32 correct{};

	for (u32 i{}; i < testingImages.numberOfItems; ++i) 
	{
		u32 batchSize{ std::min(BATCH_SIZE, testingImages.numberOfItems - i) };
		Eigen::Map<const Eigen::VectorX<u8>> correctLabels{testingLabels.data.data(), batchSize};
		Eigen::VectorX<u8> predictions{makePredictions(images.row(i))};

		correct += (predictions.array() == correctLabels.array()).count();
	}

	std::println("accuracy: {:2f}", correct / testingImages.numberOfItems);

}

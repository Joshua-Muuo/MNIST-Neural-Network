#include "Core/neural_network.hpp"
#include "Core/dense_layer.hpp"
#include "Core/relu_layer.hpp"
#include "Core/softmax.hpp"
#include <chrono>
#include <iostream>
#include <iterator>
#include <memory>
#include <print>
#include "Core/utils.hpp"

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

Matrix NeuralNetwork::forward(Matrix input) const
{
	// input.setRandom();
	for (auto& ptr : mLayers)
		input = ptr->forward(input);

	return input;
}

Matrix NeuralNetwork::forwardWithoutSoftmax(Matrix input) const
{
	if (mLayers.empty()) return input;
	for (auto it{mLayers.begin()}; it != mLayers.end() - 1; ++it)
		input = (*it)->forward(input);

	return input;
}

void NeuralNetwork::backward(Matrix predOutputs, std::span<f32> labels) const
{
	
	Eigen::Map<RowMajorMatrix> labelMatrixMap{labels.data(), predOutputs.rows(), 10};
	Matrix dOutput{predOutputs - labelMatrixMap};
	for (auto it{mLayers.rbegin()}; it != mLayers.rend(); ++it)
		dOutput = (*it)->backward(dOutput);
	
}

void NeuralNetwork::train(u32 epochs)
{
	Utils::Data trainingImages{Utils::loadImages("data/train-images-idx3-ubyte.gz")};
	Utils::Data trainingLabels{Utils::loadLabels("data/train-labels-idx1-ubyte.gz")};

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
			augmentData(batch);
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

void NeuralNetwork::augmentData(std::span<f32> data)
{
	usize images{data.size() / Utils::PIXELS_PER_IMAGE};
	for (usize image{}; image < images; ++image)
	{
		std::span<f32> currentImage{
			data.subspan(image * Utils::PIXELS_PER_IMAGE, Utils::PIXELS_PER_IMAGE)};
		rotateImage(currentImage);
		translateImage(currentImage);
	}
}
	
void NeuralNetwork::rotateImage(std::span<f32> data)
{
	std::vector<f32> newImage(784, 0.0f);
	f32 angle{Utils::rand(-10.0f, 10.0f)};

	f32 width{28}, height{28};
	const f32 cx = (width - 1) / 2.0f;
    const float cy = (height - 1) / 2.0f;

    const float cosA = std::cos(angle);
    const float sinA = std::sin(angle);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float dx = x - cx;
            float dy = y - cy;

            float sx = cosA * dx + sinA * dy + cx;
            float sy = -sinA * dx + cosA * dy + cy;

            int x0 = static_cast<int>(std::floor(sx));
            int y0 = static_cast<int>(std::floor(sy));

            int x1 = x0 + 1;
            int y1 = y0 + 1;

            if (x0 < 0 || y0 < 0 ||
                x1 >= width || y1 >= height) {
                continue;
            }

            float a = sx - x0;
            float b = sy - y0;

            newImage[y * 28+ x] =
                (1 - a) * (1 - b) * data[x0 + y0 * 28] +
                a * (1 - b) * data[y0 * 28 + x1] +
                (1 - a) * b * data[y1 * 28 + x0] +
                a * b * data[y1 * 28 + x1];
        }

	}

	std::copy(newImage.begin(), newImage.end(), data.begin());
}

void NeuralNetwork::translateImage(std::span<f32> data)
{
	std::vector<f32> newImage(784, 0.0f);
	i32 dx{Utils::rand(-2, 2)}, dy{Utils::rand(-2, 2)};
	for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < 28 &&
                ny >= 0 && ny < 28) {
                newImage[ny*28 +  nx] = data[y * 28 + x];
            }
        }
    }

	std::copy(newImage.begin(), newImage.end(), data.begin());
}

void NeuralNetwork::applyGradientDescent()
{
	const f32 learningRate{0.01f};
	for (auto& layer : mLayers)
		layer->applyGradientDescent(learningRate);
}

u32 NeuralNetwork::makePrediction(const Matrix& mat) const
{
	
	Matrix predOutput{forwardWithoutSoftmax(mat)};
	Eigen::Index predictedDigit{};

	predOutput.row(0).maxCoeff(&predictedDigit);
	return predictedDigit;
}

u32 NeuralNetwork::makePrediction(std::span<f32, 784> data) const
{
	Eigen::Map<RowMajorMatrix> mat{data.data(), 1, Utils::PIXELS_PER_IMAGE};
	Matrix predOutput{forwardWithoutSoftmax(mat)};
	Eigen::Index predictedDigit{};

	predOutput.row(0).maxCoeff(&predictedDigit);
	return predictedDigit;
}

Eigen::VectorXi NeuralNetwork::makePredictions(const Matrix& mat) const
{

	Matrix predOutputs{forwardWithoutSoftmax(mat)};
	Eigen::VectorXi maxIndices{mat.rows()};

	for (usize i{}; i < mat.rows(); ++i)
		predOutputs.row(i).maxCoeff(&maxIndices[i]);
	
	return maxIndices;
}

void NeuralNetwork::saveData() const
{
	for (const auto& layer : mLayers)
	{
		layer->saveData();
	}
}

void NeuralNetwork::loadWeightsAndBiases(const std::string& pathToModel)
{
	DenseLayer* denseLayer1{dynamic_cast<DenseLayer*>(mLayers[0].get())};
	DenseLayer* denseLayer2{dynamic_cast<DenseLayer*>(mLayers[2].get())};

	denseLayer1->setWeights(
		Utils::loadWeights(
				pathToModel + std::format("/dense_layer_weights_{}x{}", 
					denseLayer1->rows(), denseLayer1->cols())
		));

	denseLayer1->setBiases(Utils::loadBiases(
			pathToModel + std::format("/dense_layer_biases_1x{}", denseLayer1->rows())
		));

	denseLayer2->setWeights(
		Utils::loadWeights(
				pathToModel + std::format("/dense_layer_weights_{}x{}", 
					denseLayer2->rows(), denseLayer2->cols())
		));

	denseLayer2->setBiases(Utils::loadBiases(
			pathToModel + std::format("/dense_layer_biases_1x{}", denseLayer2->rows())
		));
}

f32 NeuralNetwork::testAccuracy()
{
	Utils::Data testingImages{Utils::loadImages("data/t10k-images-idx3-ubyte.gz")};
	Utils::Data testingLabels{Utils::loadLabels("data/t10k-labels-idx1-ubyte.gz")};

	std::vector<f32> imagesBuffer{Utils::loadBatchImages(testingImages.data, 0, testingImages.numberOfItems)};

	Eigen::Map<RowMajorMatrix> images{imagesBuffer.data(), testingImages.numberOfItems, 784};

	u32 randImageIndex{Utils::printImage(imagesBuffer)};

	std::println("prediction: {}", makePrediction(images.row(randImageIndex)));
	std::println("actual: {}\n\n", testingLabels.data[randImageIndex]);

	u32 correctPredictions{};
	for (u32 i{}; i < testingImages.numberOfItems; i += BATCH_SIZE) 
	{
		u32 batchSize{ std::min(BATCH_SIZE, testingImages.numberOfItems - i) };
		Eigen::Map<const Eigen::VectorX<u8>> correctLabels{testingLabels.data.data() + i, batchSize};
		auto batch{images.middleRows(i, batchSize)};
		Eigen::VectorX<u8> predictions{makePredictions(batch).cast<u8>()};
		correctPredictions += (predictions.array() == correctLabels.array()).count();
	}

	const f32 accuracy{static_cast<f32>(correctPredictions) / testingImages.numberOfItems};
	return accuracy;
}

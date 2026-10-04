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
	void forward();
	void backward();

	std::vector<std::unique_ptr<Layer>> mLayers;
};

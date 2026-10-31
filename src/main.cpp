#include "Core/neural_network.hpp"
#include <print>
#include <iostream>
#include <Eigen/Dense>
#include "Graphics/window.hpp"

int main(i32 argc, const char* argv[])
{
	// Eigen::MatrixXd test(2,2);
	std::vector<std::string> args{argv + 1, argv + argc};
	NeuralNetwork nn{784, 100, 10};
	const std::string modelLocation{"models/model1"};
	nn.loadWeightsAndBiases(modelLocation);
	//
	// u64 duration{};
	if (args.size() <= 1)
	{
		const f32 accuracy{nn.testAccuracy()};
		std::println("accuracy of {:2f}%", 100 * accuracy);
	}
	else
	{
		for (usize i{}; i < args.size(); ++i)
		{
			const std::string& arg{args[i]};
			if (arg == "--train")
			{
				i32 epochs{std::stoi(args[1])};
				std::println("Training with {} epochs", epochs);
				nn.train(epochs);
				f32 accuracy{nn.testAccuracy()};
				std::println("accuracy: {}%", 100 * accuracy);
			}
		}
	}
	Window window{"Test"};
	window.run();
	
	// TODO: Create Util namespace holding global random number generator
	// TODO: Download MNIST data to project
	// TODO: Create functions to load MNIST data 
	
	// nn.train(4);


	return 0;
}

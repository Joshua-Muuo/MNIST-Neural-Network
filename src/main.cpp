#include "neural_network.hpp"
#include <print>
#include <iostream>
#include <Eigen/Dense>

#include "utils.hpp"

int main()
{
	std::println("Hello World!");
	Eigen::MatrixXd test(2,2);
	std::cout << "test matrix is " << test << '\n';

	NeuralNetwork nn{784, 100, 10};
	// TODO: Create Util namespace holding global random number generator
	// TODO: Download MNIST data to project
	// TODO: Create functions to load MNIST data 
	nn.train();


	return 0;
}

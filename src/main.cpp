#include <print>
#include <iostream>
#include <Eigen/Dense>

int main()
{
	std::println("Hello World!");
	Eigen::MatrixXd test(2,2);
	std::cout << "test matrix is " << test << '\n';
	test << 1, 2, 3;
	return 0;
}

#include <Eigen/Core>
#include <random>
#include <optional>
#include <vector>
#include "types.hpp"
#include <span>
#include <Eigen/Dense>

using Matrix = Eigen::MatrixXf; 
using RowMajorMatrix = Eigen::Matrix<f32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace Utils
{
	struct Data
	{
		u32 magicNumber{};	
		u32 numberOfItems{};
		std::optional<u32> rows, cols{};
		std::vector<u8> data{};
	};

	extern std::mt19937 gen;

	// returns a random f32 value in the interval [0.0, 1.0)
	f32 randFloat();

	// returns a random value in the interval [0, a)
	template<typename T>
	T rand(T a)
	{
		return static_cast<T>(a * randFloat());
	}

	// returns a random value in the interval [a, b)
	template<typename T>
	T rand(T a, T b)	
	{
		return static_cast<T>((b - a) * randFloat()) + a;
	}


	// Matrix matrixDot(Matrix& a, const Matrix b);
	Data loadImage(const std::string& filename);
	Data loadLabel(const std::string& filename);
	std::vector<f32> loadBatchImages(std::span<u8> data, u32 startingIndex = 0, u32 batchSize = 100);
	std::vector<f32> loadBatchLabels(std::span<u8> data, u32 startingIndex = 0, u32 batchSize = 100);

	Matrix loadWeights(const std::string& filename);
	void saveWeights(const Eigen::MatrixXf& weights, std::string filename = "");

	Eigen::RowVectorXf loadBiases(const std::string& filename);
	void saveBiases(const Eigen::RowVectorXf& biases, std::string filename = "");

	
	u32 printImage(std::span<u8> data, std::optional<u32> index = std::nullopt);
	u32 printImage(std::span<f32> data, std::optional<u32> index = std::nullopt);
}

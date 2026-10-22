#include "utils.hpp"
#include "zconf.h"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <print>
#include <random>
#include <zlib.h>

namespace Utils
{
	std::mt19937 gen{std::random_device()()};

	f32 randFloat()
	{
		static std::uniform_real_distribution<f32> dist(0.0f, 1.0f);
		return dist(gen);
	}

	std::vector<u8> loadGZipFile(const std::string& filename)
	{	
		std::vector<u8> output, compressedFile;		

		// std::ifstream file{filename, std::ios::binary};
		// usize fileSize{std::filesystem::file_size(filename)};
		// std::println("file is {} bytes", fileSize);
		//
		// compressedFile.resize(fileSize);
		// file.read((char*)compressedFile.data(), fileSize);


		const u32 chunkSize{4096};
		output.resize(chunkSize * 4);

		gzFile file{gzopen(filename.c_str(), "r")};
		isize bytes{}, totalBytesRead{};
		i32 err;
		while (true)
		{
			if (totalBytesRead + chunkSize >= output.size())
				output.resize(output.size() + 64 * chunkSize);

			bytes = gzread(file, output.data() + totalBytesRead, chunkSize);

			if (bytes < 0)
			{
				// const char* errMsg{gzerror(file, &err)};
				// std::println(stderr, "error: {}, errCode: {}", errMsg, err);
				break;

			}
			else if (bytes == 0) break;
			totalBytesRead += bytes;

		}
		gzclose(file);
		
	

		output.resize(totalBytesRead);
		return output;
	}

	Data loadImage(const std::string& filename)
	{
		Data output;
		output.data = loadGZipFile(filename);

		// header is 16 bytes and is stored in big-endian
		const u32 HEADER_SIZE{16};

		std::span<u32> header{(u32*)output.data.data(), HEADER_SIZE / 4};
		for (u32& value : header)
			value = std::byteswap(value);


		output.magicNumber = header[0];
		output.numberOfItems = header[1];
		output.rows = header[2];
		output.cols = header[3];

		output.data.erase(output.data.begin(), output.data.begin() + HEADER_SIZE);

		return output;
	}

	Data loadLabel(const std::string& filename)
	{
		Data output;
		output.data = loadGZipFile(filename);

		// header is 8 bytes and is stored in big-endian
		const u32 HEADER_SIZE{8};

		std::span<u32> header{(u32*)output.data.data(), HEADER_SIZE};
		for (u32& value : header)
			value = std::byteswap(value);

		output.magicNumber = header[0];
		output.numberOfItems = header[1];
		output.rows = std::nullopt; 
		output.cols = std::nullopt;

		output.data.erase(output.data.begin(), output.data.begin() + HEADER_SIZE);
		return output;
	}
	
	std::vector<f32> loadBatchImages(std::span<u8> data, u32 startingIndex, u32 batchSize)
	{
		std::vector<f32> output;
		output.reserve(batchSize * 784);
		// std::copy(
		// 		data.begin() + startingIndex, 
		// 		data.begin() + startingIndex + batchSize, 
		// 		output.begin()
		// 		);
		//

		for (const u8& value : data)
			output.push_back((f32)value / 255.0f);

		return output;
	}

	std::vector<f32> loadBatchLabels(std::span<u8> data, u32 startingIndex, u32 batchSize)
	{
		std::vector<f32> output(batchSize * 10, 0.0f);

		for (usize i{}; i < batchSize; ++i)
		{
			const u8& label{data[startingIndex + i]};
			output[label + 10 * i] = 1.0f; 
		}

		return output;
	}

	u32 printImage(std::span<u8> data, std::optional<u32> index)
	{
		u32 NUMBER_OF_PIXELS{784};
		const usize numberOfImages{data.size() / NUMBER_OF_PIXELS};
		if (!index.has_value())
			index = rand(numberOfImages);

		static const char* BLACK{"\033[40m"}, *WHITE{"\033[47m"}, *RESET{"\033[0m"};
		for (usize i{NUMBER_OF_PIXELS * index.value()}; i < NUMBER_OF_PIXELS * (index.value() + 1); ++i)
		{
			if ((i) % 28 == 0 && i != 0)
				std::println("{}", RESET);
			// std::print("{}", (i - 16) % 28);
			std::print("\033[48;2;{};{};{}m  {}", data[i], data[i], data[i], RESET);
		}

		return index.value();
	}

	u32 printImage(std::span<f32> data, std::optional<u32> index)
	{
		u32 NUMBER_OF_PIXELS{784};
		const usize numberOfImages{data.size() / NUMBER_OF_PIXELS};
		if (!index.has_value())
			index = rand(numberOfImages);

		std::vector<u8> imageBuffer(NUMBER_OF_PIXELS, 0);
		for (usize i{}; i < NUMBER_OF_PIXELS; ++i)
		{
			imageBuffer[i] = 256 * data[NUMBER_OF_PIXELS * index.value() + i];
		}

		printImage(imageBuffer, 0);
		return index.value();
	}

	Matrix loadWeights(const std::string& filename)
	{
		
		std::ifstream file{filename, std::ios::binary};
		
		i64 magicNumber{};
		file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));

		// if (magicNumber != 0x12344321) return;

		i64 rows, cols;
		file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
		file.seekg(sizeof(i64), std::ios_base::cur);

		file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
		file.seekg(sizeof(i64), std::ios_base::cur);

		std::vector<f32> buffer(rows * cols * sizeof(f32), 0.0f);
		file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

		Eigen::Map<RowMajorMatrix> mapMatrix(buffer.data(), rows, cols);
		return mapMatrix;
}

	void saveWeights(const Eigen::MatrixXf& weights, std::string filename)
	{
		if (filename == "")
			filename = std::format("dense_layer_weights_{}x{}", weights.rows(), weights.cols());
		std::ofstream file{"layers/" + filename, std::ios::binary};		

		std::array<i64, 6> data{0x12344321, weights.rows(), 0u, weights.cols(), 0u};
		file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(i64));
		// file << 0x12344321 << weights.rows() << 0u << weights.cols() << 0u;

		std::println("saving weights to layers/{}", filename);

		file.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(f32));
	}

	Eigen::RowVectorXf loadBiases(const std::string& filename)
	{
		std::ifstream file{filename, std::ios::binary};
		
		i64 magicNumber{};
		file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));

		// if (magicNumber != 0x12344321) return;

		i64 rows, cols;
		file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
		file.seekg(sizeof(i64), std::ios_base::cur);

		file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
		file.seekg(sizeof(i64), std::ios_base::cur);

		std::vector<f32> buffer(rows * cols * sizeof(f32), 0.0f);
		file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

		Eigen::Map<RowMajorMatrix> mapMatrix(buffer.data(), rows, cols);
		return mapMatrix;
	}

	void saveBiases(const Eigen::RowVectorXf& biases, std::string filename)
	{
		if (filename.empty())
			filename = std::format("dense_layer_biases_{}x{}", biases.rows(), biases.cols());
		std::ofstream file{"layers/" + filename, std::ios::binary};		

		std::array<i64, 5> data{0x78900987, biases.rows(), 0u, biases.cols(), 0u};
		file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(i64));
		// file << 0x78900987 << biases.rows() << 0u << biases.cols() << 0u;

		file.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(f32));
	}
}

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
	const u32 PIXELS_PER_IMAGE{784};

	f32 randFloat()
	{
		static std::uniform_real_distribution<f32> dist(0.0f, 1.0f);
		return dist(gen);
	}

	f32 normalDistFloat()
	{
		static std::normal_distribution<f32> dist(0.0f, 1.0f);
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


		const u32 chunkSize{4096u};
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

	Data loadImages(const std::string& filename)
	{
		Data output;
		output.data = loadGZipFile(filename);

		// header is 16 bytes and is stored in big-endian
		const u32 HEADER_BYTES{16u};
		const u32 HEADER_UINT_COUNT{HEADER_BYTES / sizeof(u32)};


		std::span<u32> header{reinterpret_cast<u32*>(output.data.data()), HEADER_UINT_COUNT};
		for (u32& value : header)
			value = std::byteswap(value);

		output.magicNumber = header[0];
		output.numberOfItems = header[1];
		output.rows = header[2];
		output.cols = header[3];

		output.data.erase(output.data.begin(), output.data.begin() + HEADER_BYTES);
		return output;
	}

	Data loadLabels(const std::string& filename)
	{
		Data output;
		output.data = loadGZipFile(filename);

		// header is 8 bytes and is stored in big-endian

		const u32 HEADER_BYTES{8u};
		const u32 HEADER_UINT_COUNT{HEADER_BYTES / sizeof(u32)};

		std::span<u32> header{reinterpret_cast<u32*>(output.data.data()), HEADER_UINT_COUNT};
		for (u32& value : header)
			value = std::byteswap(value);

		output.magicNumber   = header[0];
		output.numberOfItems = header[1];
		output.rows	         = std::nullopt; 
		output.cols          = std::nullopt;

		output.data.erase(output.data.begin(), output.data.begin() + HEADER_BYTES);
		return output;
	}
	
	std::vector<f32> loadBatchImages(std::span<u8> data, u32 startingIndex, u32 batchSize)
	{
		std::vector<f32> output;
		output.reserve(batchSize * PIXELS_PER_IMAGE);

		usize start{startingIndex * PIXELS_PER_IMAGE};
		usize numberOfPixels{batchSize * PIXELS_PER_IMAGE};
		if (start + numberOfPixels > data.size())
			numberOfPixels = data.size() - start;

		// std::println("startingIndex is {}, batchSize is {}", startingIndex, batchSize);
		// std::println("start is {}, num pixels is {}", start, numberOfPixels);
		// std::println("size of span is {}", data.size());
		std::span<u8> batch{data.subspan(start, numberOfPixels)};
		for (const u8& value : batch)
			output.push_back(value / 255.0f);

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
		const usize numberOfImages{data.size() / PIXELS_PER_IMAGE};
		if (!index.has_value())
			index = rand(numberOfImages);

		static const char* BLACK{"\033[40m"}, *WHITE{"\033[47m"}, *RESET{"\033[0m"};
		for (usize i{PIXELS_PER_IMAGE * index.value()}; i < PIXELS_PER_IMAGE * (index.value() + 1); ++i)
		{
			if ((i) % 28 == 0 && i != 0)
				std::println("{}", RESET);
			// std::print("{}", (i - 16) % 28);
			std::print("\033[48;2;{};{};{}m  {}", data[i], data[i], data[i], RESET);
		}

		std::println();

		return index.value();
	}

	u32 printImage(std::span<f32> data, std::optional<u32> index)
	{
		const usize numberOfImages{data.size() / PIXELS_PER_IMAGE};
		if (!index.has_value())
		{
			index = rand(numberOfImages);
			std::println("generating rand value of printImage, rand {}", rand(numberOfImages));
		}

		std::vector<u8> imageBuffer(PIXELS_PER_IMAGE, 0);

		std::println("starting index for printImage is {}", *index);
		for (usize i{}; i < PIXELS_PER_IMAGE; ++i)
		{
			imageBuffer[i] = 256 * data[(*index) * PIXELS_PER_IMAGE + i];
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

		Eigen::Map<Matrix> mapMatrix(buffer.data(), rows, cols);
		return mapMatrix;
}

	void saveWeights(const Eigen::MatrixXf& weights, std::string filename)
	{
		if (filename == "")
			filename = std::format("dense_layer_weights_{}x{}", weights.rows(), weights.cols());
		std::ofstream file{"layers/" + filename, std::ios::binary};		

		std::array<i64, 5> data{0x12344321, weights.rows(), 0u, weights.cols(), 0u};
		file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(i64));
		// file << 0x12344321 << weights.rows() << 0u << weights.cols() << 0u;

		std::println("saving weights to models/{}", filename);

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

		Eigen::Map<Matrix> mapMatrix(buffer.data(), rows, cols);
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

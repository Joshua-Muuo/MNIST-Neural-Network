#include "utils.hpp"
#include <filesystem>
#include <iterator>
#include <print>
#include <random>
#include <zlib.h>
#include <fstream>
#include <arpa/inet.h>

namespace Utils
{
	std::mt19937 gen{std::random_device()()};

	f32 randFloat()
	{
		static std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);
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

		std::span<u32> header{(u32*)output.data.data(), HEADER_SIZE};
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
		u32* num;

		// header is 8 bytes and is stored in big-endian
		const u32 HEADER_SIZE{8};

		std::span<u32> header{(u32*)output.data.data(), HEADER_SIZE};
		for (u32& value : header)
			value = std::byteswap(value);

		output.data.erase(output.data.begin(), output.data.begin() + HEADER_SIZE);
		return output;
	}
}

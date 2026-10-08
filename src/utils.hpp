#include <random>
#include <vector>
#include "types.hpp"

namespace Utils
{
	extern std::mt19937 gen;

	// returns a random f32 value between (-1.0, 1.0)
	f32 randFloat();

	std::vector<u8> loadImage(const std::string& filename);
	std::vector<u8> loadLabel(const std::string& filename);
}

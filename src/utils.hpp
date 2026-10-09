#include <random>
#include <optional>
#include <vector>
#include "types.hpp"

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

	// returns a random f32 value between (-1.0, 1.0)
	f32 randFloat();

	Data loadImage(const std::string& filename);
	Data loadLabel(const std::string& filename);
}

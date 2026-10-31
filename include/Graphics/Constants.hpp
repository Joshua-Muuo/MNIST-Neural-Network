#include "types.hpp"
#include <string_view>
namespace Graphics
{
	inline constexpr u32 PIXEL_SIZE{16u}, ROWS{28u}, COLS{28u};
	inline constexpr u32 PREDICTION_SPACE{400u}, MARGIN_BOTTOM{100u};
	// inline constexpr u32 PREDICTION_MARGIN{200}, MARGIN_BOTTOM{100};
	inline constexpr u32 CANVAS_WIDTH{COLS * PIXEL_SIZE};
	inline constexpr u32 CANVAS_HEIGHT{ROWS * PIXEL_SIZE};
	inline constexpr u32 WIDTH{CANVAS_WIDTH + PREDICTION_SPACE};
	inline constexpr u32 HEIGHT{CANVAS_HEIGHT + MARGIN_BOTTOM};

	inline constexpr u32 MARGIN{20};
	inline constexpr u32 END_OF_GRID{COLS * PIXEL_SIZE}, BOTTOM_OF_GRID{ROWS * PIXEL_SIZE};

	constexpr std::string_view FONT_LOCATION{"data/arcade.ttf"};
};

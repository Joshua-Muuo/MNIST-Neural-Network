#pragma once

#include "types.hpp"
#include <string_view>
namespace Graphics
{
	inline constexpr i32 PIXEL_SIZE{20u}, ROWS{28u}, COLS{28u};
	inline constexpr i32 PREDICTION_SPACE{400u}, MARGIN_BOTTOM{100u};
	// inline constexpr i32 PREDICTION_MARGIN{200}, MARGIN_BOTTOM{100};
	inline constexpr i32 CANVAS_WIDTH{COLS * PIXEL_SIZE};
	inline constexpr i32 CANVAS_HEIGHT{ROWS * PIXEL_SIZE};
	inline constexpr i32 NUM_OF_PIXELS{ROWS * COLS};
	inline constexpr i32 WIDTH{CANVAS_WIDTH + PREDICTION_SPACE};
	inline constexpr i32 HEIGHT{CANVAS_HEIGHT + MARGIN_BOTTOM};

	inline constexpr i32 MARGIN{20};
	inline constexpr i32 END_OF_GRID{COLS * PIXEL_SIZE}, BOTTOM_OF_GRID{ROWS * PIXEL_SIZE};

	constexpr std::string_view FONT_LOCATION{"data/arcade.ttf"};
};

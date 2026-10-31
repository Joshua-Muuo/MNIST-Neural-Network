#include <SDL3/SDL.h>
#include "Constants.hpp"
class Canvas
{
public:
	Canvas(u32 width = Graphics::CANVAS_WIDTH, u32 height = Graphics::CANVAS_HEIGHT);
	
	void handleEvent(const SDL_Event& event);

private:
	i32 mBrushSize{2};
	u32 mWidth{}, mHeight{};
};

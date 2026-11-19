#include "Graphics/canvas.hpp"
#include "Graphics/Constants.hpp"
#include "Graphics/window.hpp"
#include <print>
#include <cmath>


namespace  
{
	f32 gaussianBlur(u32 x0, u32 y0, u32 x, u32 y)
	{
		return std::exp(- ((x - x0)*(x - x0)/2.f + (y - y0)*(y - y0) / 2.f) );
	}
}
Canvas::Canvas(u32 width, u32 height, SDL_Renderer* renderer):
mWidth{width}, mHeight{height}, mRenderer{renderer}
{
}

Canvas::Canvas(u32 width, u32 height):
mWidth{width}, mHeight{height}
{
}

void Canvas::setRenderer(SDL_Renderer* renderer) { mRenderer = renderer; }
void Canvas::drawLine(SDL_Point p1, SDL_Point p2, SDL_Color color)
{
	SDL_SetRenderDrawColor(mRenderer, color.r, color.g, color.b, color.a);
	SDL_RenderLine(mRenderer, p1.x, p1.y, p2.x, p2.y);
}

void Canvas::drawLine(SDL_Point p1, SDL_Point p2)
{
	SDL_SetRenderDrawColor(mRenderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
	SDL_RenderLine(mRenderer, p1.x, p1.y, p2.x, p2.y);
}

void Canvas::handleEvent(const SDL_Event& event)
{
	static bool mouseDown{false};
	if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN || mouseDown)
	{
		SDL_FPoint p{.x = static_cast<f32>(event.button.x) / Graphics::CANVAS_WIDTH  * Graphics::COLS, 
					 .y = static_cast<f32>(event.button.y) / Graphics::CANVAS_HEIGHT * Graphics::ROWS};

		gaussianDab(p);

		mouseDown = true;
	}

	if (event.type == SDL_EVENT_MOUSE_BUTTON_UP) mouseDown = false;
}

void Canvas::gaussianDab(SDL_FPoint p0)
{
	const f32 WHITE{1.0f};
	i32 radius{static_cast<i32>(1.5f * mBrushRadius)};
	f32 sigma{radius / 3.0f}, twoSigmaSq{2.0f * sigma * sigma};

	f32 flow{0.5f};

	i32 minX{std::max<i32>(0, p0.x - radius)},
		minY{std::max<i32>(0, p0.y - radius)},
		maxX{std::min<i32>(Graphics::COLS - 1, p0.x + radius)},
		maxY{std::min<i32>(Graphics::ROWS - 1, p0.y + radius)};


	for (i32 y{minY}; y <= maxY; ++y)
	{
		f32 dy{y + 0.5f - p0.y}, dySq{dy * dy};
		for (i32 x{minX}; x <= maxX; ++x)
		{
			f32 dx{(x + 0.5f) - p0.x};
			float distSq{dx * dx + dySq};

			if (distSq > radius * radius) continue;

			// Gaussian falloff weight (0.0 to 1.0)
			f32 alpha{std::exp(-distSq / twoSigmaSq) * flow};
			if (alpha <= 0.001f) continue;

			// Linear interpolation: dst = dst + alpha * (src - dst)
			usize idx{static_cast<usize>(y) * Graphics::COLS + x};
			f32 dstVal{static_cast<float>(mCanvas[idx])};
			f32 outVal{dstVal + alpha * (WHITE - dstVal)};

			mCanvas[idx] = outVal;
			if (x == p0.x && y == p0.y)
			{
				std::println("Looping over center, color {}", outVal);
			}

		}
	}
}

void Canvas::drawStroke(SDL_FPoint p1, SDL_FPoint p2)
{
	f32 dx{p2.x - p1.x}, dy{p2.y - p1.y};
	f64 distance{std::hypot(dx, dy)};

	i32 steps{static_cast<i32>(distance / 0.15f)};
	for (i32 i{}; i <= steps; ++i)
	{
		f32 t{static_cast<f32>(i) / steps};
		gaussianDab({p1.x + dx * t, p1.y + dy * t});
	}
}

void Canvas::drawGrid()
{
	// draw vertical lines
	// f32 x{Graphics::PIXEL_SIZE};
	// for (u32 row{}; row < Graphics::ROWS; ++row)
	// {
	// 	drawLine({x, 0}, {x, Graphics::BOTTOM_OF_GRID});
	// 	x += Graphics::PIXEL_SIZE;
	// }
	//
	// // draw horizontal lines
	// f32 y{Graphics::PIXEL_SIZE};
	// for (u32 col{}; col < Graphics::COLS; ++col)
	// {
	// 	drawLine({0, y}, {Graphics::END_OF_GRID, y});
	// 	y += Graphics::PIXEL_SIZE;
	// }

	static SDL_FRect square{.x = 0, .y = 0, .w = Graphics::PIXEL_SIZE - 1, .h = Graphics::PIXEL_SIZE - 1};
	for (i32 x{}; x < Graphics::COLS; ++x)
	{
		drawLine({x * Graphics::PIXEL_SIZE, 0}, {x * Graphics::PIXEL_SIZE, Graphics::BOTTOM_OF_GRID});
		for (i32 y{}; y < Graphics::ROWS; ++y)
		{
			if (x == 0)
			{
				drawLine(
						{0, y * Graphics::PIXEL_SIZE}, 
						{Graphics::END_OF_GRID, y * Graphics::PIXEL_SIZE}
					);
			}

			u32 color{static_cast<u32>(mCanvas[Graphics::COLS * y + x] * 255)};
			SDL_SetRenderDrawColor(mRenderer, color, color, color, SDL_ALPHA_OPAQUE);
			square.x = x * Graphics::PIXEL_SIZE;
			square.y = y * Graphics::PIXEL_SIZE;


			SDL_RenderFillRect(mRenderer, &square);
		}
	}
	drawLine({Graphics::CANVAS_WIDTH, 0}, {Graphics::CANVAS_WIDTH, Graphics::BOTTOM_OF_GRID});
	drawLine({0, Graphics::CANVAS_HEIGHT}, {Graphics::END_OF_GRID, Graphics::CANVAS_HEIGHT});

}

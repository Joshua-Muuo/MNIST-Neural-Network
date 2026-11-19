#pragma once

#include <SDL3/SDL.h>
#include <array>
#include <vector>

#include "Constants.hpp"
class Canvas
{
public:
	Canvas(u32 width, u32 height, SDL_Renderer* renderer);
	Canvas(u32 width, u32 height);
	

	void toggleErase();
	void drawLine(SDL_Point p1, SDL_Point p2, SDL_Color color);
	void drawLine(SDL_Point p1, SDL_Point p2);
	void drawGrid();
	void setRenderer(SDL_Renderer* renderer);
	void handleEvent(const SDL_Event& event);
	const std::array<f32, Graphics::NUM_OF_PIXELS>& getCanvas() const;

private:
	void gaussianDab(SDL_FPoint p0);
	void drawStroke(SDL_FPoint p1, SDL_FPoint p2);
	void clearCanvas();
private:
	i32 mBrushRadius{2};
	bool mEraseMode{false};
	u32 mWidth{}, mHeight{};
	std::array<f32, Graphics::NUM_OF_PIXELS> mCanvas{};
	SDL_Renderer* mRenderer{};
};

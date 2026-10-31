#pragma once
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3_ttf/SDL_ttf.h>
#include <string>
#include "types.hpp"

class Window
{
public:
	Window(const std::string& title);
	~Window();

	void run();

public:
	static constexpr u32 PIXEL_SIZE{16}, ROWS{28}, COLS{28};
	static constexpr u32 PREDICTION_SPACE{400}, MARGIN_BOTTOM{100};
	// static constexpr u32 PREDICTION_MARGIN{200}, MARGIN_BOTTOM{100};
	static constexpr u32 WIDTH{COLS * PIXEL_SIZE + PREDICTION_SPACE};
	static constexpr u32 HEIGHT{ROWS * PIXEL_SIZE + MARGIN_BOTTOM};
	static constexpr u32 MARGIN{20};
	static constexpr u32 END_OF_GRID{COLS * PIXEL_SIZE}, BOTTOM_OF_GRID{ROWS * PIXEL_SIZE};

private:
	void drawLine(SDL_FPoint p1, SDL_FPoint p2, SDL_Color color);
	void drawLine(SDL_FPoint p1, SDL_FPoint p2);
	void drawGrid();
	void drawPrediction();
private:
	SDL_Window* mWindow;
	SDL_Renderer* mRenderer;
	TTF_TextEngine* mTextEngine;
	TTF_Font* mPredictionFont;
};

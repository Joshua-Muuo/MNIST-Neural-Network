#pragma once
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3_ttf/SDL_ttf.h>
#include <string>
#include "types.hpp"
#include "canvas.hpp"

class Window
{
public:
	Window(const std::string& title);
	~Window();

	void run();

public:

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
	Canvas mCanvas;
};

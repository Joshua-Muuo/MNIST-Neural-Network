#pragma once
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3_ttf/SDL_ttf.h>
#include <string>
#include "types.hpp"
#include "Core/neural_network.hpp"
#include "canvas.hpp"
#include "imgui.h"
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_sdlrenderer3.h"


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
	void drawScreen();
	void drawInstructions();
	void initImGUI();
private:
	SDL_Window* mWindow;
	SDL_Renderer* mRenderer;
	TTF_TextEngine* mTextEngine;
	TTF_Font* mPredictionFont;
	TTF_Font* mInstructionFont;
	Canvas mCanvas;
	NeuralNetwork mNN{784, 100, 10};
};

#include "window.hpp"
#include <SDL3/SDL_pixels.h>
#include <SDL3/SDL_render.h>
#include <print>
#include <format>
#include <string_view>


constexpr std::string_view FONT_LOCATION{"data/arcade.ttf"};
Window::Window(const std::string& title)
{
	if (!SDL_Init(SDL_INIT_VIDEO))
	{
		throw std::runtime_error(std::format("Couldn't initialize SDL: {}", SDL_GetError()));
	}

	if (!SDL_CreateWindowAndRenderer(title.c_str(), WIDTH, HEIGHT, 0, &mWindow, &mRenderer))
	{
		throw std::runtime_error(std::format("Couldn't create Window/Renderer: {}", SDL_GetError()));
	}

	TTF_Init();
	mTextEngine = TTF_CreateRendererTextEngine(mRenderer);
	mPredictionFont = TTF_OpenFont(FONT_LOCATION.data(), 24.0f); // path, point size

	std::println("Creating window with dimension {}x{}", WIDTH, HEIGHT);
}

Window::~Window()
{
	SDL_DestroyRenderer(mRenderer);
    SDL_DestroyWindow(mWindow); //
    SDL_Quit();
}

void Window::run()
{
	SDL_Event event;

	bool running{true};
	while (running)
	{
		while (SDL_PollEvent(&event))
		{
			if (event.type == SDL_EVENT_QUIT) 
				running = false;
		}

		SDL_SetRenderDrawColor(mRenderer, 0, 0, 0, 255);
        SDL_RenderClear(mRenderer);

		drawGrid();
		drawPrediction();
		SDL_RenderPresent(mRenderer);
	}
}

void Window::drawPrediction()
{
	static TTF_Text* text{TTF_CreateText(mTextEngine, mPredictionFont, "PREDICTION", 0)};
	TTF_SetTextColor(text, 128, 128, 128, SDL_ALPHA_OPAQUE); 
	TTF_DrawRendererText(text, END_OF_GRID + MARGIN, MARGIN);

	static TTF_Text* prediction{TTF_CreateText(mTextEngine, mPredictionFont, "3", 0)};
	TTF_DrawRendererText(prediction, END_OF_GRID + PREDICTION_SPACE / 2, 4 * MARGIN);
}
	
void Window::drawLine(SDL_FPoint p1, SDL_FPoint p2, SDL_Color color)
{
	SDL_SetRenderDrawColor(mRenderer, color.r, color.g, color.b, color.a);
	SDL_RenderLine(mRenderer, p1.x, p1.y, p2.x, p2.y);
}

void Window::drawLine(SDL_FPoint p1, SDL_FPoint p2)
{
	SDL_SetRenderDrawColor(mRenderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
	SDL_RenderLine(mRenderer, p1.x, p1.y, p2.x, p2.y);
}

void Window::drawGrid()
{
	f32 x{PIXEL_SIZE};
	// draw vertical lines
	for (u32 row{}; row < ROWS; ++row)
	{
		drawLine({x, 0}, {x, BOTTOM_OF_GRID});
		x += PIXEL_SIZE;
	}

	f32 y{PIXEL_SIZE};
	for (u32 col{}; col < COLS; ++col)
	{
		drawLine({0, y}, {END_OF_GRID, y});
		y += PIXEL_SIZE;
	}
}

#include "Graphics/window.hpp"
#include <SDL3/SDL_pixels.h>
#include <SDL3/SDL_render.h>
#include <print>
#include <format>
#include <string_view>


Window::Window(const std::string& title):
mCanvas{Graphics::CANVAS_WIDTH, Graphics::CANVAS_HEIGHT}
{
	if (!SDL_Init(SDL_INIT_VIDEO))
	{
		throw std::runtime_error(std::format("Couldn't initialize SDL: {}", SDL_GetError()));
	}

	if (!SDL_CreateWindowAndRenderer(
				title.c_str(), Graphics::WIDTH, Graphics::HEIGHT, 0, &mWindow, &mRenderer
		))
	{
		throw std::runtime_error(
				std::format("Couldn't create Window/Renderer: {}", SDL_GetError())
		);
	}
	mCanvas.setRenderer(mRenderer);

	TTF_Init();
	mTextEngine = TTF_CreateRendererTextEngine(mRenderer);
	mPredictionFont = TTF_OpenFont(Graphics::FONT_LOCATION.data(), 24.0f); // path, point size

	std::println("Creating window with dimension {}x{}", Graphics::WIDTH, Graphics::HEIGHT);
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

			mCanvas.handleEvent(event);
		}

		SDL_SetRenderDrawColor(mRenderer, 0, 0, 0, 255);
        SDL_RenderClear(mRenderer);

		mCanvas.drawGrid();
		drawPrediction();
		SDL_RenderPresent(mRenderer);
	}
}

void Window::drawPrediction()
{
	static TTF_Text* text{TTF_CreateText(mTextEngine, mPredictionFont, "PREDICTION", 0)};
	TTF_SetTextColor(text, 128, 128, 128, SDL_ALPHA_OPAQUE); 
	TTF_DrawRendererText(text, Graphics::END_OF_GRID + Graphics::MARGIN, Graphics::MARGIN);

	static TTF_Text* prediction{TTF_CreateText(mTextEngine, mPredictionFont, "3", 0)};
	TTF_DrawRendererText(prediction, Graphics::END_OF_GRID + Graphics::PREDICTION_SPACE / 2, 4 * Graphics::MARGIN);
}
	


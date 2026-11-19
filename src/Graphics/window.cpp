#include "Graphics/window.hpp"
#include "Graphics/Constants.hpp"
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
	mInstructionFont= TTF_OpenFont(Graphics::FONT_LOCATION.data(), 18.0f); // path, point size

	initImGUI();
	const std::string modelLocation{"models/model1"};
	mNN.loadWeightsAndBiases(modelLocation);
	std::println("Creating window with dimension {}x{}", Graphics::WIDTH, Graphics::HEIGHT);
}

Window::~Window()
{
	ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
	
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

		ImGui_ImplSDLRenderer3_NewFrame();
		ImGui_ImplSDL3_NewFrame();
		ImGui::NewFrame();

		mCanvas.drawGrid();
		drawPrediction();

		ImGui::Render();
		drawScreen();
		drawInstructions();

		SDL_RenderPresent(mRenderer);	
	}
}

void Window::drawPrediction()
{
	static TTF_Text* text{TTF_CreateText(mTextEngine, mPredictionFont, "PREDICTION", 0)};
	TTF_SetTextColor(text, 128, 128, 128, SDL_ALPHA_OPAQUE); 
	TTF_DrawRendererText(text, Graphics::END_OF_GRID + Graphics::MARGIN, Graphics::MARGIN);

	auto canvas{mCanvas.getCanvas()};
	u32 pred{mNN.makePrediction(canvas)};
	std::string predText{std::format("{}", pred)};
	std::println("prediction is {}", pred);
	TTF_Text* prediction = TTF_CreateText(mTextEngine, mPredictionFont, predText.c_str(), 0);
	TTF_DrawRendererText(prediction, Graphics::END_OF_GRID + Graphics::PREDICTION_SPACE / 2, 4 * Graphics::MARGIN);

}
	
void Window::drawScreen()
{
	ImGui_ImplSDLRenderer3_RenderDrawData(
        ImGui::GetDrawData(),
        mRenderer
    );
}

void Window::drawInstructions()
{
	TTF_Text* clearInstructs{TTF_CreateText(mTextEngine, mInstructionFont, 
			"C : clear\nE : erase", 0)};
	TTF_DrawRendererText(clearInstructs, Graphics::MARGIN, Graphics::CANVAS_HEIGHT + Graphics::MARGIN);

}

void Window::initImGUI()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable keyboard controls
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    ImGui_ImplSDL3_InitForSDLRenderer(mWindow, mRenderer);
    ImGui_ImplSDLRenderer3_Init(mRenderer);

}

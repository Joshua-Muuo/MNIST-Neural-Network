cmake_minimum_required(VERSION 3.15...4.4)
set(CMAKE_CXX_STANDARD 23)

project(NeuralNetwork VERSION 1.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(FetchContent)

set(SRC_FILES 
	src/main.cpp 
	src/Graphics/window.cpp
	src/Graphics/canvas.cpp
	src/Core/dense_layer.cpp
	src/Core/neural_network.cpp
	src/Core/relu_layer.cpp
	src/Core/utils.cpp
	src/Core/softmax.cpp
)


FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        5.0.0  
    GIT_SHALLOW    TRUE   
    FIND_PACKAGE_ARGS CONFIG
)

set(EIGEN_BUILD_DOC OFF CACHE INTERNAL "")
set(EIGEN_BUILD_PKGCONFIG OFF CACHE INTERNAL "")
set(BUILD_TESTING OFF CACHE INTERNAL "")

FetchContent_MakeAvailable(Eigen3)

FetchContent_Declare(
    zlib
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG        v1.3.2 
)

set(ZLIB_BUILD_TESTING OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(zlib)


FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        docking   
)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(
	SDL3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
	GIT_TAG main
)
FetchContent_MakeAvailable(SDL3)

FetchContent_Declare(
	SDL3_ttf
    GIT_REPOSITORY https://github.com/libsdl-org/SDL_ttf
	GIT_TAG main
)
FetchContent_MakeAvailable(SDL3_ttf)

# string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPER_BUILD_TYPE)
# if(UPPER_BUILD_TYPE STREQUAL "RELEASE")
# 	add_compile_options(-O3 -march=native -ffast-math -DNDEBUG -DEIGEN_NO_DEBUG)
# 	message("Hello")
# endif()

add_executable(NeuralNetwork ${SRC_FILES})
# target_include_directories(NeuralNetwork
# ${sdl3_ttf_SOURCE_DIR}/include)
# find_package(glm CONFIG REQUIRED)
# find_package(glfw3 REQUIRED)
#
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdlrenderer3.cpp
)

target_include_directories(imgui PUBLIC 
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

if(NOT TARGET ZLIB::ZLIB)
    if(TARGET zlibstatic)
        add_library(ZLIB::ZLIB ALIAS zlibstatic)
    elseif(TARGET zlib)
        add_library(ZLIB::ZLIB ALIAS zlib)
    endif()
endif()



target_link_libraries(NeuralNetwork 
	Eigen3::Eigen 
	ZLIB::ZLIB 
	SDL3::SDL3 
	imgui
	SDL3_ttf::SDL3_ttf)	

target_include_directories(NeuralNetwork 
	PUBLIC 
	${CMAKE_CURRENT_SOURCE_DIR}/include)
# target_link_libraries(NeuralNetwork PUBLIC glm::glm glfw imgui)	

target_compile_options(NeuralNetwork PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
    $<$<CONFIG:Release>:-DNDEBUG>
    $<$<CONFIG:Release>:-DEIGEN_NO_DEBUG>
)

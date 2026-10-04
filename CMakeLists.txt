cmake_minimum_required(VERSION 3.15...4.4)
set(CMAKE_CXX_STANDARD 23)

project(NeuralNetwork VERSION 1.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(FetchContent)

set(SRC_FILES 
	src/main.cpp 
)

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        5.0.0  
    GIT_SHALLOW    TRUE   
    FIND_PACKAGE_ARGS CONFIG
)
FetchContent_MakeAvailable(Eigen3)

set(EIGEN_BUILD_DOC OFF CACHE INTERNAL "")
set(EIGEN_BUILD_PKGCONFIG OFF CACHE INTERNAL "")
set(BUILD_TESTING OFF CACHE INTERNAL "")

# FetchContent_Declare(
#     imgui
#     GIT_REPOSITORY https://github.com/ocornut/imgui.git
#     GIT_TAG        docking   
# )
# FetchContent_MakeAvailable(imgui)

add_executable(NeuralNetwork ${SRC_FILES})
# target_include_directories(NeuralNetwork PRIVATE
	# ${CMAKE_CURRENT_SOURCE_DIR}/include)
# find_package(glm CONFIG REQUIRED)
# find_package(glfw3 REQUIRED)
#
# add_library(imgui STATIC
#     ${imgui_SOURCE_DIR}/imgui.cpp
#     ${imgui_SOURCE_DIR}/imgui_demo.cpp
#     ${imgui_SOURCE_DIR}/imgui_draw.cpp
#     ${imgui_SOURCE_DIR}/imgui_tables.cpp
#     ${imgui_SOURCE_DIR}/imgui_widgets.cpp
#     ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
#     ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
# )
#
# target_include_directories(imgui PUBLIC 
#     ${imgui_SOURCE_DIR}
#     ${imgui_SOURCE_DIR}/backends
# )
#



target_link_libraries(NeuralNetwork Eigen3::Eigen)	
# target_link_libraries(NeuralNetwork PUBLIC glm::glm glfw imgui)	

cmake_minimum_required(VERSION 3.17)
project(
  cubemap_converter
  VERSION 0.1
  DESCRIPTION "Convert cubemaps to native images."
  LANGUAGES C CXX)
set(CMAKE_COLOR_MAKEFILE ON)

message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")

# Add third party code
add_subdirectory(dependencies)

set(GLAD_SOURCES dependencies/glad/src/gl.c)
set(PROJECT_SOURCES source/main.cc source/gl_utils.cc)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES} ${GLAD_SOURCES})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME} PRIVATE 
  "${CMAKE_SOURCE_DIR}/dependencies/stb"
  "${CMAKE_SOURCE_DIR}/dependencies/glad/include"
  )

# Add dependencies
add_dependencies(${PROJECT_NAME} glfw fmt-header-only CLI11 glm)
target_link_libraries(${PROJECT_NAME} glfw fmt-header-only opengl32 CLI11 glm)

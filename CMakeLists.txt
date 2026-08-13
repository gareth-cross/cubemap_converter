cmake_minimum_required(VERSION 3.31)
project(
  cubemap_converter
  VERSION 0.1
  DESCRIPTION "Convert cubemaps to native images."
  LANGUAGES C CXX)

find_package(OpenGL REQUIRED)
find_package(PNG REQUIRED)

add_subdirectory(dependencies)
add_subdirectory(shaders)

add_executable(
  cubemap_converter
  source/main.cc source/gl_utils.cc source/gl_utils.hpp source/images.cc
  source/images.hpp dependencies/glad/src/gl.c)
target_compile_features(cubemap_converter PUBLIC cxx_std_20)
set_property(TARGET cubemap_converter PROPERTY C_STANDARD 11)

target_include_directories(
  cubemap_converter
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/stb"
          "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glad/include"
          "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/scope_guard")

if(MSVC)
  target_compile_options(cubemap_converter PRIVATE /W4 /WX /D_USE_MATH_DEFINES
                                                   /wd4244)
else()
  target_compile_options(cubemap_converter PRIVATE -Wall -Wextra -pedantic
                                                   -Werror)
endif()

target_link_libraries(
  cubemap_converter
  CLI11::CLI11
  fmt::fmt-header-only
  glfw
  glm
  opengl32
  PNG::PNG
  gx::assertions
  shaders)

include(GNUInstallDirs)
install(
  TARGETS cubemap_converter
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

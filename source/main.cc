#include <filesystem>
#include <string>

#define GL_SILENCE_DEPRECATION
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_FAILURE_STRINGS

#include <glad/gl.h>

#include <GLFW/glfw3.h>
#include <fmt/format.h>
#include <stb_image.h>
#include <CLI/CLI.hpp>


int main(int argc, char** argv) {
  CLI::App app{"Preview data"};

  std::string input_path{};
  app.add_option("-i,--input-path", input_path, "Path to the input images.")->required();
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }
  return 0;
}

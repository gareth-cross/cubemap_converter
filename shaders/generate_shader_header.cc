#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include <fmt/format.h>
#include <CLI/CLI.hpp>

constexpr std::string_view shader_template = R"glsl(// Machine generated file - do not modify.
// Generated from: {}
#pragma once
#include <string_view>

namespace shaders {{

constexpr std::string_view {} = R"({})";

}} // namespace shaders
)glsl";

int main(const int argc, char** argv) {
  CLI::App app{"Generate shader header"};
  std::string input{};
  std::string output{};
  try {
    app.add_option("-i,--input", input, "Path to the input file.")->required();
    app.add_option("-o,--output", output, "Path to the output file.")->required();
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  std::fstream input_stream(input, std::ios::in | std::ios::binary);
  if (!input_stream.good()) {
    return 1;
  }

  std::fstream output_stream(output, std::ios::out);
  output_stream << fmt::format(
      shader_template, input, std::filesystem::path(input).stem().string(),
      std::string(std::istreambuf_iterator<char>(input_stream), std::istreambuf_iterator<char>()));
  output_stream.flush();
  if (!output_stream.good()) {
    return 1;
  }
  return 0;
}

#include <array>
#include <filesystem>
#include <functional>
#include <optional>
#include <variant>

#include <glad/gl.h>

#include <GLFW/glfw3.h>
#include <fmt/format.h>
#include <CLI/CLI.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "assertions.hpp"
#include "gl_utils.hpp"
#include "images.hpp"

// Include all the shaders, which we generate from the files in `shaders/*.glsl`
#include "shaders/fragment_cubemap.hpp"
#include "shaders/fragment_display.hpp"
#include "shaders/vertex.hpp"

static void glfw_error_callback(int error, const char* description) {
  fmt::print("GLFW error. Code = {}, Message = {}\n", error, description);
}

// Group together all the input arguments.
struct ProgramArgs {
  std::string input_path;
  std::size_t index;
  std::string table_path;
  int table_width;
  int table_height;
  bool enable_gl_debug;
  std::string valid_mask_path;
};

// Parse program arts, or fail and return exit code.
std::variant<ProgramArgs, int> ParseProgramArgs(int argc, char** argv) {
  CLI::App app{"Cubemap converter"};
  ProgramArgs args{};
  try {
    app.add_option("-i,--input-path", args.input_path, "Path to the input dataset.")->required();
    app.add_option("--index", args.index, "Index of the image to display.")->required();
    app.add_option("-t,--remap-table", args.table_path, "Path to the remap table.")->required();
    app.add_option("--width", args.table_width, "Width of the native image.")->required();
    app.add_option("--height", args.table_height, "Height of the native image.")->required();
    app.add_option("--debug", args.enable_gl_debug, "Enable OpenGL debug log (v4.3 or higher).");
    app.add_option("--mask", args.valid_mask_path, "Optional valid mask image (png).");
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  } catch (const CLI::Error& e) {
    fmt::print("Some other exception: {}", e.what());
    return 1;
  }
  return args;
}

gl_utils::Texture2D LoadValidMask(const std::string& mask_path, const int table_width, const int table_height) {
  gl_utils::Texture2D texture{};
  if (mask_path.empty()) {
    // No mask, just put a white image in (valid everywhere).
    images::SimpleImage white_image{table_width, table_height, 1, images::ImageDepth::Bits8};
    std::fill(white_image.data.begin(), white_image.data.end(), 255);
    texture.Fill(white_image);
  } else {
    std::optional<images::SimpleImage> mask_image = images::LoadPng(mask_path, images::ImageDepth::Bits8);
    ASSERT(mask_image.has_value(), "Could not load valid mask from: {}", mask_path);
    ASSERT(mask_image->width == table_width && mask_image->height == table_height,
           "Remap table and valid mask do not share the same dimensions. mask = [{}, {}], table = [{}, {}]",
           mask_image->width, mask_image->height, table_width, table_height);
    texture.Fill(*mask_image);
  }
  return texture;
}

void ExecuteMainLoop(const ProgramArgs& args, GLFWwindow* const window) {
  ASSERT(args.table_width > 0 && args.table_height > 0, "Dimensions must be positive: w={}, h={}", args.table_width,
         args.table_height);

  // Enable seamless cubemaps
  glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

  const std::filesystem::path dataset{args.input_path};

  // Load the remap table.
  const images::SimpleImage remap_table_img =
      images::LoadRawFloatImage(args.table_path, args.table_width, args.table_height, 3);

  // Copy remap table to GPU:
  const gl_utils::Texture2D remap_table{remap_table_img};

  // Load the valid mask
  const gl_utils::Texture2D valid_mask = LoadValidMask(args.valid_mask_path, args.table_width, args.table_height);

  // Load all the RGB images:
  const std::vector<std::optional<images::SimpleImage>> faces =
      images::LoadCubemapImages(dataset, args.index, 0, images::CubemapType::Rgb);

  // Create a cube-map:
  gl_utils::TextureCube cube{};
  for (int face = 0; face < 6; ++face) {
    const std::optional<images::SimpleImage>& image_opt = faces[face];
    ASSERT(image_opt.has_value(), "Failed to load cubemap face: {}", face);
    cube.Fill(face, *image_opt);
  }

  // Create shader for building native image:
  const gl_utils::ShaderProgram cubemap_shader_program =
      gl_utils::CompileShaderProgram(shaders::vertex, shaders::fragment_cubemap);

  // Create shader for displaying the native image in the UI:
  const gl_utils::ShaderProgram display_program =
      gl_utils::CompileShaderProgram(shaders::vertex, shaders::fragment_display);

  // Create projection matrix:
  const glm::mat4x4 projection = glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
  cubemap_shader_program.SetMatrixUniform("projection", projection);
  display_program.SetMatrixUniform("projection", projection);

  // The rotation from a DirectX camera to an unreal camera: (UE cam has +x forward, per their pawn convention).
  constexpr glm::fquat unreal_cam_R_directx_cam = glm::fquat{0.5f, 0.5f, 0.5f, 0.5f};
  cubemap_shader_program.SetMatrixUniform("cubemap_R_camera", glm::mat3_cast(unreal_cam_R_directx_cam));

  // Create a quad in a buffer.
  // The viewport is configured so that bottom left is [0, 0] and top right is [1, 1].
  // Vertices are packed as [x, y, z, u, v].
  // clang-format off
  const std::array<float, 5 * 4> vertices = {
      1.0f, 1.0f, 0.0f,    1.0f, 1.0f,  // top right
      1.0f, 0.0f, 0.0f,    1.0f, 0.0f,  // bottom right
      0.0f, 0.0f, 0.0f,    0.0f, 0.0f,  // bottom left
      0.0f, 1.0f, 0.0f,    0.0f, 1.0f,  // top left
  };
  const std::array<unsigned int, 6> triangles = {
      1, 0, 3,
      3, 2, 1
  };
  // clang-format on

  // Create buffers
  GLuint vertex_buffer, vertex_array, index_buffer;
  glGenVertexArrays(1, &vertex_array);
  glGenBuffers(1, &vertex_buffer);
  glGenBuffers(1, &index_buffer);
  glBindVertexArray(vertex_array);

  // Send vertex data.
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

  // Send triangle data
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * triangles.size(), &triangles[0], GL_STATIC_DRAW);

  // Specify how vertices are arranged in the buffer:
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
  glEnableVertexAttribArray(0);

  // Specify how texture coordinates are arranged in the buffer:
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // Unbind
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  // Create a frame buffer to render into:
  GLuint frame_buffer;
  glGenFramebuffers(1, &frame_buffer);
  ASSERT(frame_buffer);
  glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer);

  // Color buffer for the fbo:
  GLuint color_buffer_texture;
  glGenTextures(1, &color_buffer_texture);
  ASSERT(color_buffer_texture);
  glBindTexture(GL_TEXTURE_2D, color_buffer_texture);

  // Create a 32-bit float render target for the output:
  const int texture_width = args.table_width;
  const int texture_height = args.table_height;
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, texture_width, texture_height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_buffer_texture, 0);
  const GLenum fbo_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  ASSERT(fbo_status == GL_FRAMEBUFFER_COMPLETE, "FBO is not complete, status = {}", fbo_status);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  // Cull clockwise back-faces
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  glFrontFace(GL_CCW);

  // Main loop
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // render to texture
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer);
    glViewport(0, 0, texture_width, texture_height);
    glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, remap_table.Handle());

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cube.Handle());

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, valid_mask.Handle());

    // Tell the shader what we are rendering:
    cubemap_shader_program.SetUniformInt("remap_table", 0);
    cubemap_shader_program.SetUniformInt("input_cube", 1);
    cubemap_shader_program.SetUniformInt("valid_mask", 2);
    cubemap_shader_program.SetUniformInt("is_depth", false);

    glUseProgram(cubemap_shader_program.Handle());
    glBindVertexArray(vertex_array);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    // done drawing to fbo
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Set up the main viewport
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // These variables ensure we render with the correct aspect ratio:
    display_program.SetUniformVec2("viewport_dims", glm::vec2(display_w, display_h));
    display_program.SetUniformVec2("image_dims", glm::vec2(texture_width, texture_height));
    display_program.SetUniformInt("image", 0);

    // Draw the image to the screen:
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, color_buffer_texture);

    glUseProgram(display_program.Handle());
    glBindVertexArray(vertex_array);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    glfwSwapBuffers(window);
  }

  // Read contents of the RGB buffer back:
#if 0
  images::SimpleImage output{texture_width, texture_height, 1, images::ImageDepth::Bits16};

  glBindTexture(GL_TEXTURE_2D, color_buffer_texture);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_SHORT, &output.data[0]);
#else
  images::SimpleImage output{texture_width, texture_height, 3, images::ImageDepth::Bits8};

  glBindTexture(GL_TEXTURE_2D, color_buffer_texture);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, &output.data[0]);
#endif

  //  std::vector<uint16_t> float_values;
  //  float_values.resize(output.data.size() / sizeof(uint16_t));
  //  std::memcpy(&float_values[0], &output.data[0], output.data.size());
  //
  //  const auto min = *std::min_element(float_values.begin(), float_values.end());
  //  const auto max = *std::max_element(float_values.begin(), float_values.end());
  //  fmt::print("min = {}, max = {}\n", min, max);

  // Save it out
  fmt::print("Saving image out!");
  images::WritePng("output.png", output, true);
}

// Callback to update viewport.
void WindowSizeCallback(GLFWwindow* const window, int, int) {
  int display_w, display_h;
  glfwGetFramebufferSize(window, &display_w, &display_h);
  glViewport(0, 0, display_w, display_h);
}

int Run(const ProgramArgs& args) {
  // Setup window
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) {
    fmt::print("Failed to initialize GLFW.\n");
    return 1;
  }

  // GL 3.3
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  // Create window with graphics context
  GLFWwindow* const window = glfwCreateWindow(1280, 720, "Cubemap converter", nullptr, nullptr);
  if (window == nullptr) {
    fmt::print("Failed to create GLFW window\n");
    return 1;
  }
  glfwMakeContextCurrent(window);

  // print the version of OpenGL:
  const int glad_version = gladLoadGL(glfwGetProcAddress);
  fmt::print("Using OpenGL {}.{} (GLAD generator = v{})\n", GLAD_VERSION_MAJOR(glad_version),
             GLAD_VERSION_MINOR(glad_version), GLAD_GENERATOR_VERSION);

  // Enable vsync
  glfwSwapInterval(1);
  glfwSetWindowSizeCallback(window, WindowSizeCallback);

  // print errors
  if (args.enable_gl_debug) {
    gl_utils::EnableDebugOutput(glad_version);
  }

  // Render until the window closes:
  ExecuteMainLoop(args, window);

  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}

int main(int argc, char** argv) {
  // Parse args or bail.
  const std::variant<ProgramArgs, int> args_or_error = ParseProgramArgs(argc, argv);
  if (args_or_error.index() == 1) {
    return std::get<int>(args_or_error);
  }
  return Run(std::get<ProgramArgs>(args_or_error));
}

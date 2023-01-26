#include <array>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <variant>

#define GL_SILENCE_DEPRECATION

#include <glad/gl.h>

#include <GLFW/glfw3.h>
#include <fmt/format.h>
#include <CLI/CLI.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "assertions.hpp"
#include "gl_utils.hpp"
#include "images.hpp"

static void glfw_error_callback(int error, const char* description) {
  fmt::print("GLFW error. Code = {}, Message = {}\n", error, description);
}

// Group together all the input arguments.
struct ProgramArgs {
  std::string input_path;
  std::size_t index;
};

// Parse program arts, or fail and return exit code.
std::variant<ProgramArgs, int> ParseProgramArgs(int argc, char** argv) {
  CLI::App app{"Cubemap converter"};
  ProgramArgs args{};
  app.add_option("-i,--input-path", args.input_path, "Path to the input dataset.")->required();
  app.add_option("--index", args.index, "Index of the image to display.")->required();

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }
  return args;
}

// Get appropriate OpenGL for the given face index.
GLenum TargetForFace(int face) {
  constexpr std::array<GLenum, 6> faces = {
      GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X, GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
      GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
  };
  ASSERT(face >= 0 && face < 6, "Invalid face: {}", face);
  return faces[face];
}

static const std::string_view vertex_source = R"(
#version 330 core
layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 uv;

uniform mat4 projection;

out vec2 TexCoords;

void main() {
  gl_Position = projection * vec4(pos.x, pos.y, pos.z, 1.0);
  TexCoords = uv;
}
)";

static const std::string_view fragment_source_cubemap_render = R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

void main() {
  FragColor = vec4(TexCoords.x, TexCoords.y, 0.0f, 1.0f);
}
)";

// Program for previewing the image in the viewport.
static const std::string_view fragment_source_image_render = R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

// Texture we are going to display.
uniform sampler2D image;

void main() {
  vec3 rgb = texture(image, TexCoords).xyz;
  FragColor = vec4(rgb.x, rgb.y, rgb.z, 1.0f);
}
)";

void ExecuteMainLoop(const ProgramArgs& args, GLFWwindow* const window) {
  // Enable seamless cubemaps
  glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

  // Load a cubemap
  GLuint texture_handle{0};
  glGenTextures(1, &texture_handle);
  ASSERT(texture_handle, "Failed to create texture handle");
  glBindTexture(GL_TEXTURE_CUBE_MAP, texture_handle);

  const std::filesystem::path dataset{args.input_path};

  // Load all the RGB images:
  const std::vector<std::optional<images::SimpleImage>> faces =
      images::LoadCubemapImages(dataset, args.index, 0, images::CubemapType::Rgb);

  for (int face = 0; face < 6; ++face) {
    const std::optional<images::SimpleImage>& image_opt = faces[face];
    ASSERT(image_opt.has_value(), "Failed to load cubemap face");

    const images::SimpleImage& image = *image_opt;
    if (face == 0) {
      // Allocate cubemap:
      glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_RGB32F, image.width, image.height);
    }

    // Copy face to GPU:
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glTexSubImage2D(TargetForFace(face), 0, 0, 0, image.width, image.height, GL_RGB, GL_UNSIGNED_BYTE, &image.data[0]);
  }

  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  // Create shader for building native image:
  const gl_utils::ShaderProgram cubemap_shader_program =
      gl_utils::CompileShaderProgram(vertex_source, fragment_source_cubemap_render);

  // Create shader for
  const gl_utils::ShaderProgram display_program =
      gl_utils::CompileShaderProgram(vertex_source, fragment_source_image_render);

  // Create projection matrix:
  const glm::mat4x4 projection = glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
  cubemap_shader_program.SetMatrixUniform("projection", projection);
  display_program.SetMatrixUniform("projection", projection);

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

  const int texture_width = 640;
  const int texture_height = 480;
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_width, texture_height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
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
  images::SimpleImage output{texture_width, texture_height, 3, images::ImageDepth::Bits8};

  glBindTexture(GL_TEXTURE_2D, color_buffer_texture);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, &output.data[0]);

  // Save it out
  fmt::print("Saving image out!");
  images::WritePng("output.png", output, true);
}

int Run(const ProgramArgs& args) {
  // Setup window
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) {
    fmt::print("Failed to initialize GLFW.\n");
    return 1;
  }

  // GL 3.3
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
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
  fmt::print("Using OpenGL {}.{}\n", GLAD_VERSION_MAJOR(glad_version), GLAD_VERSION_MINOR(glad_version));

  // Enable vsync
  glfwSwapInterval(1);

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

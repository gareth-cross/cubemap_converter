#include <array>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
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
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  } catch (const CLI::Error& e) {
    fmt::print("Some other exception: {}", e.what());
    return 1;
  }
  return args;
}

static const std::string_view vertex_source = R"(
#version 330 core
layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 uv;

// Projection matrix of the viewport.
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

// Rotation matrix from typical camera to DirectX Cubemap (what we exported).
uniform mat3 cubemap_R_camera;

// The remap table.
uniform sampler2D remap_table;

// The cubemap.
uniform samplerCube input_cube;

void main() {
  // Lookup the unit vector:
  vec3 v_cam = normalize(texture(remap_table, TexCoords).xyz);
  // Sample the cube-map:
  vec3 color = texture(input_cube, cubemap_R_camera * v_cam).rgb;
  FragColor = vec4(color, 1.0f);
}
)";

// Program for previewing the image in the viewport.
static const std::string_view fragment_source_image_render = R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

// Texture we are going to display.
uniform sampler2D image;

// Viewport dims in pixels.
uniform vec2 viewport_dims;

// Image dims in pixels.
uniform vec2 image_dims;

void main() {
  // determine scale factor to fit the image in the viewport
  float scale_factor =
      min(viewport_dims.x / image_dims.x, viewport_dims.y / image_dims.y);

  // scale image dimensions to fit
  vec2 scaled_image_dims = image_dims * scale_factor;

  // compute offset to center:
  vec2 image_origin = (viewport_dims - scaled_image_dims) / vec2(2.0, 2.0);

  // compute coords inside the viewport bounding box [0 -> viewport_dims]
  vec2 viewport_coords = TexCoords * viewport_dims;

  // transform viewport coordinates into image coords (normalized)
  vec2 image_coords = (viewport_coords - image_origin) / scaled_image_dims;

  // are they inside the box?
  bvec2 inside_upper_bound = lessThan(image_coords, vec2(1.0, 1.0));
  bvec2 inside_lower_bound = greaterThan(image_coords, vec2(0.0, 0.0));
  float mask = float(inside_upper_bound.x && inside_upper_bound.y &&
                     inside_lower_bound.x && inside_lower_bound.y);

  vec3 rgb = texture(image, image_coords).xyz;
  FragColor = vec4(rgb.x * mask, rgb.y * mask, rgb.z * mask, 1.0f);
}
)";

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
      gl_utils::CompileShaderProgram(vertex_source, fragment_source_cubemap_render);

  // Create shader for
  const gl_utils::ShaderProgram display_program =
      gl_utils::CompileShaderProgram(vertex_source, fragment_source_image_render);

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

  const int texture_width = args.table_width;
  const int texture_height = args.table_height;
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

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, remap_table.Handle());
    cubemap_shader_program.SetUniformInt("remap_table", 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cube.Handle());
    cubemap_shader_program.SetUniformInt("input_cube", 1);

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
  images::SimpleImage output{texture_width, texture_height, 3, images::ImageDepth::Bits8};

  glBindTexture(GL_TEXTURE_2D, color_buffer_texture);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, &output.data[0]);

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

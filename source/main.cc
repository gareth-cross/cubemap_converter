#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <variant>

#define GL_SILENCE_DEPRECATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_NO_FAILURE_STRINGS

#include <glad/gl.h>

#include <GLFW/glfw3.h>
#include <fmt/format.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <CLI/CLI.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "assertions.hpp"

static void glfw_error_callback(int error, const char* description) {
  fmt::print("GLFW error. Code = {}, Message = {}\n", error, description);
}

// Very simple image type.
template <typename T>
struct SimpleImage {
  // Only two types supported here:
  static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t>);

  explicit SimpleImage(const std::filesystem::path& path) : data{nullptr, &stbi_image_free} {
    // Delegate to either 16bit or 8bit implementation:
    const std::string path_str = path.string();
    if constexpr (std::is_same_v<T, uint16_t>) {
      data.reset(stbi_load_16(path_str.c_str(), &width, &height, &components, 0));
    } else {
      data.reset(stbi_load(path_str.c_str(), &width, &height, &components, 0));
    }
    ASSERT(data.get(), "Failed to load image at path: {}", path.string());
  }

  [[nodiscard]] bool is_valid() const { return data.get() != nullptr; }

  std::unique_ptr<T[], void (*)(void*)> data{};
  int width{0};
  int height{0};
  int components{0};
};

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

static const char* vertex_source = R"(
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

static const char* fragment_source = R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

void main() {
  FragColor = vec4(TexCoords.x, TexCoords.y, 0.0f, 1.0f);
}
)";

// Compile and link the shader.
GLuint CompileShaderProgram() {
  const GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex_shader, 1, &vertex_source, nullptr);
  glCompileShader(vertex_shader);

  // check if vertex shader compiled:
  GLint success;
  std::string compiler_log;
  compiler_log.resize(1024);
  glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertex_shader, static_cast<GLsizei>(compiler_log.size()), nullptr, compiler_log.data());
    ASSERT(success, "Failed to compile vertex shader. Reason: {}", compiler_log);
  }

  // fragment shader
  const GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader, 1, &fragment_source, nullptr);
  glCompileShader(fragment_shader);

  // check for shader compile errors
  glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragment_shader, static_cast<GLsizei>(compiler_log.size()), nullptr, compiler_log.data());
    ASSERT(success, "Failed to compile fragment shader. Reason: {}", compiler_log);
  }

  // link shaders
  GLuint program = glCreateProgram();
  glAttachShader(program, vertex_shader);
  glAttachShader(program, fragment_shader);
  glLinkProgram(program);

  // check for linking errors
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(program, static_cast<GLsizei>(compiler_log.size()), nullptr, compiler_log.data());
    ASSERT(success, "Failed to link shader. Reason: {}", compiler_log);
  }
  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);
  return program;
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
  GLFWwindow* window = glfwCreateWindow(1280, 720, "Cubemap converter", nullptr, nullptr);
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

  // Enable seamless cubemaps
  glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

  // Load a cubemap
  GLuint texture_handle{0};
  glGenTextures(1, &texture_handle);
  ASSERT(texture_handle, "Failed to create texture handle");
  glBindTexture(GL_TEXTURE_CUBE_MAP, texture_handle);

  const std::filesystem::path dataset{args.input_path};
  for (int face = 0; face < 6; ++face) {
    // Load the image
    const std::filesystem::path input_path =
        dataset / "image" / "camera00" / fmt::format("{:08}_{:02}.png", args.index, face);
    const SimpleImage<uint8_t> image{input_path};

    if (face == 0) {
      // Allocate cubemap:
      glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_RGB32F, image.width, image.height);
    }

    // Copy face to GPU:
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glTexSubImage2D(TargetForFace(face), 0, 0, 0, image.width, image.height, GL_RGB, GL_UNSIGNED_BYTE,
                    image.data.get());
  }

  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  // Create shader for displaying cubemap
  const GLuint shader_program = CompileShaderProgram();

  // Create projection matrix:
  const glm::mat4x4 projection = glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
  glUseProgram(shader_program);
  const GLint projection_uniform = glGetUniformLocation(shader_program, "projection");
  ASSERT(projection_uniform != -1, "Failed to find uniform");
  glUniformMatrix4fv(projection_uniform, 1, GL_FALSE, &projection[0][0]);

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

    glUseProgram(shader_program);
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

    //    glUseProgram(shader_program);
    //    glBindVertexArray(vertex_array);
    //    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glfwSwapBuffers(window);
  }

  // Read contents of the RGB buffer back:
  std::vector<uint8_t> rgb_data(texture_width * texture_height * 3);
  glBindTexture(GL_TEXTURE_2D, color_buffer_texture);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, &rgb_data[0]);

  // Save it out
  // TODO: Flip the order here.
  fmt::print("Saving image out!");
  const int success = stbi_write_png("output.png", texture_width, texture_height, 3, &rgb_data[0], texture_width * 3);
  ASSERT(success, "Failed to write output png");

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

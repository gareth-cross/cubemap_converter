// Copyright 2023 Gareth Cross
#include <array>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <optional>
#include <queue>
#include <variant>

#include <glad/gl.h>

#include <GLFW/glfw3.h>
#include <fmt/format.h>
#include <CLI/CLI.hpp>

#include "assertions.hpp"
#include "gl_utils.hpp"
#include "images.hpp"
#include "timing.hpp"

// Include all the shaders, which we generate from the files in `shaders/*.glsl`
#include "shaders/fragment_display.hpp"
#include "shaders/fragment_oversampled_cubemap.hpp"
#include "shaders/vertex.hpp"

static void glfw_error_callback(int error, const char* description) {
  fmt::print("GLFW error. Code = {}, Message = {}\n", error, description);
}

// Group together all the input arguments.
struct ProgramArgs {
  std::string input_path;
  std::string output_path;
  std::size_t num_images;
  std::size_t camera_index;
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
    app.add_option("-o,--output-path", args.output_path, "Path to the output directory.");
    app.add_option("--num-images", args.num_images, "Num images in the dataset.")->required();
    app.add_option("-c,--camera-index", args.camera_index, "Index of the camera to render.")->required();
    app.add_option("-t,--remap-table", args.table_path, "Path to the remap table.")->required();
    app.add_option("--width", args.table_width, "Width of the native image.")->required();
    app.add_option("--height", args.table_height, "Height of the native image.")->required();
    app.add_flag("--debug", args.enable_gl_debug, "Enable OpenGL debug log (v4.3 or higher).");
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

// A poor man's thread pool.
template <typename T>
struct TaskQueue {
  explicit TaskQueue(std::size_t max) : max_items(max){};

  // Push new task into the queue.
  template <typename Function>
  void Push(Function&& func) {
    if (pending.size() == max_items) {
      std::future<T> front = std::move(pending.front());
      pending.pop();
      front.wait();
    }
    pending.push(std::async(std::launch::async, std::forward<Function>(func)));
  }

  // Clear the queue of tasks.
  void Flush() {
    while (!pending.empty()) {
      std::future<T> front = std::move(pending.front());
      pending.pop();
      front.wait();
    }
  }

  std::queue<std::future<T>> pending{};
  std::size_t max_items;
};

void CreateOrAssert(const std::filesystem::path& path) {
  std::error_code err{};
  // Recursively create directories:
  const bool created = std::filesystem::create_directories(path, err);
  ASSERT(created || !err, "Failed to create directory: `{}`. Error = {}", path.u8string(), err.message());
}

void ExecuteMainLoop(const ProgramArgs& args, GLFWwindow* const window) {
  ASSERT(args.table_width > 0 && args.table_height > 0, "Dimensions must be positive: w={}, h={}", args.table_width,
         args.table_height);

  // Path to the input directory:
  const std::filesystem::path dataset{args.input_path};

  // Create directories for the outputs:
  const std::filesystem::path output_root{args.output_path};
  const std::filesystem::path output_dir_rgb = output_root / "image" / fmt::format("camera{:02}", args.camera_index);
  const std::filesystem::path output_dir_inv_range =
      output_root / "range" / fmt::format("camera{:02}", args.camera_index);
  CreateOrAssert(output_dir_rgb);
  CreateOrAssert(output_dir_inv_range);

  // Load the remap table.
  const images::SimpleImage remap_table_img =
      images::LoadRawFloatImage(args.table_path, args.table_width, args.table_height, 3);

  // Match window to the size of the target:
  glfwSetWindowSize(window, remap_table_img.width, remap_table_img.height);

  // Copy remap table to GPU:
  const gl_utils::Texture2D remap_table{remap_table_img};

  // Load the valid mask
  const gl_utils::Texture2D valid_mask = LoadValidMask(args.valid_mask_path, args.table_width, args.table_height);

  // Create a cube-map (initially empty)
  gl_utils::TextureArray rgb_cube{};
  gl_utils::TextureArray inv_depth_cube{};

  // Create shader for building native image:
  const gl_utils::ShaderProgram cubemap_shader_program =
      gl_utils::CompileShaderProgram(shaders::vertex, shaders::fragment_oversampled_cubemap);

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

  // The size of the oversampled cubemaps, in radians:
  // TODO: Would be nice if these were read it from the dataset, instead of being hardcoded.
  cubemap_shader_program.SetUniformFloat("oversampled_fov", static_cast<float>(95.0 * M_PI / 180.0));
  cubemap_shader_program.SetUniformFloat("ue_clip_plane_meters", 0.1f);

  // A VBO w/ a quad we can draw to fill the screen:
  const gl_utils::FullScreenQuad quad{};

  // Create a frame buffer to render into:
  const int texture_width = args.table_width;
  const int texture_height = args.table_height;

  // Cull clockwise back-faces
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  glFrontFace(GL_CCW);

  // Wrap some logic we call repeatedly in the loop below:
  const auto draw_to_fbo = [&](bool is_depth) {
    glViewport(0, 0, texture_width, texture_height);
    glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, remap_table.Handle());

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D_ARRAY, is_depth ? inv_depth_cube.Handle() : rgb_cube.Handle());

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, valid_mask.Handle());

    // Tell the shader what we are rendering:
    cubemap_shader_program.SetUniformInt("remap_table", 0);
    cubemap_shader_program.SetUniformInt("input_cube", 1);
    cubemap_shader_program.SetUniformInt("valid_mask", 2);
    cubemap_shader_program.SetUniformInt("is_depth", is_depth);
    cubemap_shader_program.SetUniformInt("cubemap_dim", is_depth ? inv_depth_cube.Dimension() : rgb_cube.Dimension());

    quad.Draw(cubemap_shader_program);
  };

  // Render the cubemap to texture (first for color):
  const gl_utils::FramebufferObject rgb_fbo{texture_width, texture_height, gl_utils::FramebufferType::Color};
  const gl_utils::FramebufferObject inv_range_fbo{texture_width, texture_height,
                                                  gl_utils::FramebufferType::InverseRange};

  // We'll render to FBO then read the previous frame before queueing another read:
  gl_utils::PixelbufferQueue color_pbos{2, texture_width, texture_height, 3, images::ImageDepth::Bits8};
  gl_utils::PixelbufferQueue inv_range_pbos{2, texture_width, texture_height, 1, images::ImageDepth::Bits16};

  // Indices of images we haven't read back from the GPU yet.
  std::queue<std::size_t> queued_indices{};

  // Queue of tasks for writing images (poor man's thread pool).
  constexpr std::size_t max_writers = 8;
  TaskQueue<void> write_queue(max_writers);

  // Main loop
  timing::SimpleTimer timer{};
  std::size_t next_index = 0;
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // Load the cubemap faces:
    // TODO: We'd get better GPU usage if this was a thread pool.
    std::vector<images::SimpleImage> faces;
    timer.Record(timing::SimpleTimer::Stages::Load,
                 [&]() { faces = images::LoadCubemapImages(dataset, next_index, args.camera_index, true); });

    // Copy the RGB + depth data:
    timer.Record(timing::SimpleTimer::Stages::Unpack, [&] {
      for (int face = 0; face < 6; ++face) {
        ASSERT(!faces[face].IsEmpty(), "Failed to load RGB cubemap face: {}, index = {}", face, next_index);
        rgb_cube.Fill(face, faces[face]);
      }
      for (int face = 0; face < 6; ++face) {
        ASSERT(!faces[face].IsEmpty(), "Failed to load inverse depth cubemap face: {}, index = {}", face, next_index);
        inv_depth_cube.Fill(face, faces[face + 6]);
      }
    });

    // Render to the FBO:
    timer.Record(timing::SimpleTimer::Stages::Render, [&] {
      rgb_fbo.RenderInto([&] { draw_to_fbo(false); });
      inv_range_fbo.RenderInto([&] { draw_to_fbo(true); });
    });

    // Read it back:
    std::size_t read_index = std::numeric_limits<std::size_t>::max();
    images::SimpleImage previous_rgb_read{};
    images::SimpleImage previous_inv_range_read{};
    timer.Record(timing::SimpleTimer::Stages::Pack, [&] {
      if (color_pbos.QueueIsFull()) {
        // We've filled the queue, we need to de-queue the oldest reads:
        ASSERT(inv_range_pbos.QueueIsFull());
        previous_rgb_read = color_pbos.PopOldestRead();
        previous_inv_range_read = inv_range_pbos.PopOldestRead();
        read_index = queued_indices.front();
        queued_indices.pop();
      }
      // Queue a read for this frame:
      color_pbos.QueueReadFromFbo(rgb_fbo);
      inv_range_pbos.QueueReadFromFbo(inv_range_fbo);
      queued_indices.push(next_index);
    });

    // Write the data out (if the user specified a path).
    if (!previous_rgb_read.IsEmpty() && !args.output_path.empty()) {
      ASSERT(read_index < next_index);  //  This should be an earlier frame.
      timer.Record(timing::SimpleTimer::Stages::Write, [&] {
        write_queue.Push([read_index, rgb = std::move(previous_rgb_read),
                          inv_range = std::move(previous_inv_range_read), &output_dir_rgb, &output_dir_inv_range] {
          images::WritePng(output_dir_rgb / fmt::format("{:08}.png", read_index), rgb, true);
          images::WritePng(output_dir_inv_range / fmt::format("{:08}.png", read_index), inv_range, true);
        });
      });
    }

    // Set up the main viewport so the user sees the result:
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
    glBindTexture(GL_TEXTURE_2D, rgb_fbo.TextureHandle());
    quad.Draw(display_program);
    glBindTexture(GL_TEXTURE_2D, 0);
    glfwSwapBuffers(window);

    // Increment the index:
    if (next_index + 1 == args.num_images) {
      break;  //  We can stop.
    } else {
      ++next_index;
    }
  }

  // Complete any pending reads:
  while (!queued_indices.empty() && !args.output_path.empty()) {
    const std::size_t index = queued_indices.front();
    queued_indices.pop();
    images::SimpleImage rgb = color_pbos.PopOldestRead();
    images::SimpleImage inv_range = inv_range_pbos.PopOldestRead();
    write_queue.Push(
        [index, rgb = std::move(rgb), inv_range = std::move(inv_range), &output_dir_rgb, &output_dir_inv_range] {
          images::WritePng(output_dir_rgb / fmt::format("{:08}.png", index), rgb, true);
          images::WritePng(output_dir_inv_range / fmt::format("{:08}.png", index), inv_range, true);
        });
  }

  write_queue.Flush();  // Wait for writing to complete.
  fmt::print("Processed {} images.\n", next_index);
  timer.Summarize();
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

  // GL 4.3
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

  // vsync slows things down a fair bit
  constexpr bool vsync = false;
  glfwSwapInterval(static_cast<int>(vsync));
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

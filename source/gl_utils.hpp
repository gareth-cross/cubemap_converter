// Copyright 2023 Gareth Cross
#pragma once
#include <array>
#include <queue>

#include <glad/gl.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "assertions.hpp"
#include "images.hpp"

// A few simple utilities to manage OpenGL resources.
namespace gl_utils {

// Simple "unique_ptr" imitation for use w/ OpenGL handles.
struct OpenGLHandle {
  // Construct w/ handle and deletion logic.
  // Could do something smarter than a function pointer, but this isn't performance critical.
  OpenGLHandle(GLuint handle, void (*deleter)(GLuint) noexcept) : handle_(handle), deleter_(deleter) {
    ASSERT(deleter_, "Cannot construct with null deleter");
  }

  // Cast to bool.
  explicit operator bool() const { return handle_ != 0; }

  // Get handle.
  [[nodiscard]] GLuint Handle() const { return handle_; }

  // Non-copyable.
  OpenGLHandle& operator=(const OpenGLHandle&) = delete;
  OpenGLHandle(const OpenGLHandle&) = delete;

  // Move construct.
  OpenGLHandle(OpenGLHandle&& other) noexcept {
    handle_ = other.handle_;
    deleter_ = other.deleter_;
    other.handle_ = 0;
  }

  // Move assign.
  OpenGLHandle& operator=(OpenGLHandle&& other) noexcept {
    Cleanup();
    handle_ = other.handle_;
    deleter_ = other.deleter_;
    other.handle_ = 0;
    return *this;
  }

  ~OpenGLHandle() { Cleanup(); }

 private:
  void Cleanup() noexcept {
    if (handle_) {
      std::invoke(deleter_, handle_);
      handle_ = 0;
    }
  }

  GLuint handle_{0};
  void (*deleter_)(GLuint) noexcept {nullptr};
};

// Wrapper for shader.
struct Shader : public OpenGLHandle {
  explicit Shader(GLenum type) : OpenGLHandle(glCreateShader(type), [](GLuint x) noexcept { glDeleteShader(x); }) {}
};

// Wrapper for shader program.
struct ShaderProgram : public OpenGLHandle {
  ShaderProgram() : OpenGLHandle(glCreateProgram(), [](GLuint x) noexcept { glDeleteProgram(x); }) {}

  // Set a matrix uniform:
  void SetMatrixUniform(std::string_view name, const glm::mat4x4& matrix) const;
  void SetMatrixUniform(std::string_view name, const glm::mat3x3& matrix) const;

  // Set a scalar uniform:
  void SetUniformVec2(std::string_view name, glm::vec2 value) const;

  // Set a integer uniform
  void SetUniformInt(std::string_view name, GLint value) const;
};

// Compile and link a shader.
ShaderProgram CompileShaderProgram(std::string_view vertex_source, std::string_view fragment_source);

// Wrapper for texture.
struct Texture2D : public OpenGLHandle {
  Texture2D();
  explicit Texture2D(const images::SimpleImage& image);

  // Fill the texture from an image.
  void Fill(const images::SimpleImage& image);
};

// Wrapper for cubemap texture.
struct TextureCube : public OpenGLHandle {
  TextureCube();

  // Fill the specified face w/ the provided image.
  void Fill(int face, const images::SimpleImage& image);

 private:
  int dimension_{0};
};

// A simple "full screen quad" object.
struct FullScreenQuad {
  // Initialize all buffers.
  FullScreenQuad();

  // Render the quad w/ the provided program.
  void Draw(const ShaderProgram& program) const;

 private:
  OpenGLHandle vertex_array_;
  OpenGLHandle vertex_buffer_;
  OpenGLHandle index_buffer_;
};

enum class FramebufferType {
  // Allocate a 32-bit RGBA buffer.
  Color,
  // Allocate a 16-bit R buffer.
  InverseRange,
};

// Color-only framebuffer we render to.
struct FramebufferObject {
  // Allocate the FBO.
  FramebufferObject(int width, int height, FramebufferType type);

  // Bind, invoke, and unbind.
  template <typename Func>
  void RenderInto(Func&& func) const {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_.Handle());
    std::invoke(std::forward<Func>(func));
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  // Get texture handle for the color buffer.
  [[nodiscard]] GLuint TextureHandle() const { return texture_.Handle(); }

  // Read the contents of the color buffer back.
  [[nodiscard]] images::SimpleImage ReadContents(int channels, images::ImageDepth depth) const;

  // Read the contents of the buffer into PBO.
  void ReadIntoPixelbuffer(int channel, images::ImageDepth depth, GLuint buffer_handle) const;

 private:
  OpenGLHandle fbo_;
  OpenGLHandle texture_;
  int width_;
  int height_;
};

// Store a queue of PBOs. We can asynchronously queue reads.
struct PixelbufferQueue {
 public:
  // Allocate queue of buffers (all have to be the same type for now).
  PixelbufferQueue(std::size_t num_buffers, int width, int height, int channels, images::ImageDepth depth);

  // Check if the queue is full.
  [[nodiscard]] bool QueueIsFull() const { return pbo_pool_.empty(); }

  // True if there are pending reads to process.
  [[nodiscard]] bool HasPendingReads() const { return !pending_reads_.empty(); }

  // Queue a read from the framebuffer into the next available PBO.
  void QueueReadFromFbo(const FramebufferObject& fbo);

  // Complete the oldest read and return the resulting image.
  images::SimpleImage PopOldestRead();

 private:
  std::vector<OpenGLHandle> pbo_pool_;
  std::queue<OpenGLHandle> pending_reads_;
  int width_;
  int height_;
  int channels_;
  images::ImageDepth depth_;
};

// Get the rotation of a given cubemap face (DX convention). Returns the rotation matrix cube_R_face.
[[maybe_unused]] inline constexpr glm::fquat GetFaceRotation(const int face) {
  const auto make_quat_xyzw = [](float x, float y, float z, float w) constexpr { return glm::fquat{w, x, y, z}; };
  // clang-format off
  constexpr std::array<glm::fquat, 6> cube_R_face = {
    make_quat_xyzw(0, 0.70710678, 0, 0.70710678),   // + x
    make_quat_xyzw(0, -0.70710678, 0, 0.70710678),  // - x
    make_quat_xyzw(-0.70710678, 0, 0, 0.70710678),  // + y
    make_quat_xyzw(0.70710678, 0, 0, 0.70710678),   // - y
    make_quat_xyzw(0, 0, 0, 1),                     // + z
    make_quat_xyzw(0, 1, 0, 0),                     // - z
  };
  // clang-format on
  return cube_R_face[face];
}

// Get the matrix right_M_left. You would convert a rotation matrix using:
// right_M_left * R * left_M_right
[[maybe_unused]] inline glm::mat3x3 GetRightMLeft() {
  // These are columns:
  return glm::mat3x3{glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)};
}

// Enable printing of opengl errors.
void EnableDebugOutput(int glad_version);

}  // namespace gl_utils

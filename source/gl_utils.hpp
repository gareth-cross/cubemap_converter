// Copyright 2023 Gareth Cross
#pragma once
#include <glad/gl.h>
#include <glm/glm.hpp>

#include "assertions.hpp"

namespace images {
struct SimpleImage;
}

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

  ~OpenGLHandle() {
    if (handle_) {
      std::invoke(deleter_, handle_);
      handle_ = 0;
    }
  }

 private:
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
  explicit Texture2D(const struct images::SimpleImage& image);

  // Fill the texture from an image.
  void Fill(const struct images::SimpleImage& image);
};

// Wrapper for cubemap texture.
struct TextureCube : public OpenGLHandle {
  TextureCube();

  // Fill the specified face w/ the provided image.
  void Fill(int face, const struct images::SimpleImage& image);

 private:
  int dimension_{0};
};

// Enable printing of opengl errors.
void EnableDebugOutput();

}  // namespace gl_utils

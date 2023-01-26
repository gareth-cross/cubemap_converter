// Copyright 2023 Gareth Cross
#pragma once
#include <glad/gl.h>

#include "assertions.hpp"

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
  [[nodiscard]] GLuint handle() const { return handle_; }

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
};

// Compile and link a shader.
ShaderProgram CompileShaderProgram(std::string_view vertex_source, std::string_view fragment_source);

}  // namespace gl_utils

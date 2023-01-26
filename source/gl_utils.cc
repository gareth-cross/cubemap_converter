// Copyright 2023 Gareth Cross
#include "gl_utils.hpp"

#include <array>

namespace gl_utils {

// Compile and link the shader.
ShaderProgram CompileShaderProgram(const std::string_view vertex_source, const std::string_view fragment_source) {
  const Shader vertex_shader{GL_VERTEX_SHADER};
  ASSERT(vertex_shader, "Failed to allocate vertex shader");

  const std::array<const GLchar*, 1> vertex_source_ = {vertex_source.data()};
  glShaderSource(vertex_shader.handle(), vertex_source_.size(), vertex_source_.data(), nullptr);
  glCompileShader(vertex_shader.handle());

  // check if vertex shader compiled:
  GLint success;
  std::string compiler_log;
  compiler_log.resize(1024);
  glGetShaderiv(vertex_shader.handle(), GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertex_shader.handle(), static_cast<GLsizei>(compiler_log.size()), nullptr, compiler_log.data());
    ASSERT(success, "Failed to compile vertex shader. Reason: {}", compiler_log);
  }

  // fragment shader
  const Shader fragment_shader{GL_FRAGMENT_SHADER};
  ASSERT(fragment_shader, "Failed to allocate vertex shader");

  const std::array<const GLchar*, 1> fragment_source_ = {fragment_source.data()};
  glShaderSource(fragment_shader.handle(), fragment_source_.size(), fragment_source_.data(), nullptr);
  glCompileShader(fragment_shader.handle());

  // check for shader compile errors
  glGetShaderiv(fragment_shader.handle(), GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragment_shader.handle(), static_cast<GLsizei>(compiler_log.size()), nullptr,
                       compiler_log.data());
    ASSERT(success, "Failed to compile fragment shader. Reason: {}", compiler_log);
  }

  // link shaders
  ShaderProgram program{};
  ASSERT(program, "Failed to allocate program");
  glAttachShader(program.handle(), vertex_shader.handle());
  glAttachShader(program.handle(), fragment_shader.handle());
  glLinkProgram(program.handle());

  // check for linking errors
  glGetProgramiv(program.handle(), GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(program.handle(), static_cast<GLsizei>(compiler_log.size()), nullptr, compiler_log.data());
    ASSERT(success, "Failed to link shader. Reason: {}", compiler_log);
  }
  return program;
}

}  // namespace gl_utils

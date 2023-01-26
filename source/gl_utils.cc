// Copyright 2023 Gareth Cross
#include "gl_utils.hpp"

#include <array>

namespace gl_utils {

// TODO: Fail more gracefully maybe?
ShaderProgram CompileShaderProgram(const std::string_view vertex_source, const std::string_view fragment_source) {
  const Shader vertex_shader{GL_VERTEX_SHADER};
  ASSERT(vertex_shader, "Failed to allocate vertex shader");

  const std::array<const GLchar*, 1> vertex_source_ = {vertex_source.data()};
  glShaderSource(vertex_shader.Handle(), vertex_source_.size(), vertex_source_.data(), nullptr);
  glCompileShader(vertex_shader.Handle());

  // check if vertex shader compiled:
  GLint success;
  std::string compiler_log;
  compiler_log.resize(1024);
  glGetShaderiv(vertex_shader.Handle(), GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertex_shader.Handle(), static_cast<GLsizei>(compiler_log.size()), nullptr, compiler_log.data());
    ASSERT(success, "Failed to compile vertex shader. Reason: {}", compiler_log);
  }

  // fragment shader
  const Shader fragment_shader{GL_FRAGMENT_SHADER};
  ASSERT(fragment_shader, "Failed to allocate vertex shader");

  const std::array<const GLchar*, 1> fragment_source_ = {fragment_source.data()};
  glShaderSource(fragment_shader.Handle(), fragment_source_.size(), fragment_source_.data(), nullptr);
  glCompileShader(fragment_shader.Handle());

  // check for shader compile errors
  glGetShaderiv(fragment_shader.Handle(), GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragment_shader.Handle(), static_cast<GLsizei>(compiler_log.size()), nullptr,
                       compiler_log.data());
    ASSERT(success, "Failed to compile fragment shader. Reason: {}", compiler_log);
  }

  // link shaders
  ShaderProgram program{};
  ASSERT(program, "Failed to allocate program");
  glAttachShader(program.Handle(), vertex_shader.Handle());
  glAttachShader(program.Handle(), fragment_shader.Handle());
  glLinkProgram(program.Handle());

  // check for linking errors
  glGetProgramiv(program.Handle(), GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(program.Handle(), static_cast<GLsizei>(compiler_log.size()), nullptr, compiler_log.data());
    ASSERT(success, "Failed to link shader. Reason: {}", compiler_log);
  }
  glUseProgram(0);
  return program;
}

}  // namespace gl_utils

// Copyright 2023 Gareth Cross
#include "gl_utils.hpp"

#include <array>

#include "images.hpp"

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

static GLuint CreateTexture() {
  GLuint texture{0};
  glGenTextures(1, &texture);
  ASSERT(texture != 0, "Failed create texture handle");
  return texture;
}

Texture2D::Texture2D() : OpenGLHandle(CreateTexture(), [](GLuint x) noexcept { glDeleteTextures(1, &x); }) {}

Texture2D::Texture2D(const images::SimpleImage& image) : Texture2D() { Fill(image); }

struct TextureFormatEntry {
  constexpr TextureFormatEntry(int channels, images::ImageDepth depth, GLenum value)
      : channels(channels), depth(depth), value(value) {}

  int channels;
  images::ImageDepth depth;
  GLenum value;
};

static GLenum GetTextureRepresentation(const int channels, const images::ImageDepth depth) {
  using images::ImageDepth;
  constexpr std::array table = {TextureFormatEntry(3, ImageDepth::Bits8, GL_RGB8),
                                TextureFormatEntry(1, ImageDepth::Bits16, GL_R16),
                                TextureFormatEntry(3, ImageDepth::Bits32, GL_RGB32F)};

  const auto it = std::find_if(table.begin(), table.end(), [&](const TextureFormatEntry& entry) {
    return entry.channels == channels && entry.depth == depth;
  });
  ASSERT(it != table.end(), "Invalid channels ({}) and depth ({})", channels, static_cast<int>(depth));
  return it->value;
}

static GLenum GetTextureInputFormat(const int channels) {
  ASSERT(channels == 1 || channels == 3, "Invalid # of channels: {}", channels);
  return channels == 1 ? GL_RED : GL_RGB;
}

static constexpr GLenum GetTextureDataType(const images::ImageDepth depth) {
  switch (depth) {
    case images::ImageDepth::Bits8:
      return GL_UNSIGNED_BYTE;
    case images::ImageDepth::Bits16:
      return GL_UNSIGNED_SHORT;
    default:
      break;
  }
  ASSERT(depth == images::ImageDepth::Bits32);
  return GL_FLOAT;
}

void Texture2D::Fill(const struct images::SimpleImage& image) {
  ASSERT(Handle());
  const GLenum internal_format = GetTextureRepresentation(image.components, image.depth);

  glBindTexture(GL_TEXTURE_2D, Handle());
  glTexStorage2D(GL_TEXTURE_2D, 1, internal_format, image.width, image.height);

  // Copy data to GPU:
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image.width, image.height, GetTextureInputFormat(image.components),
                  GetTextureDataType(image.depth), &image.data[0]);

  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

}  // namespace gl_utils

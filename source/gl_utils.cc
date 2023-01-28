// Copyright 2023 Gareth Cross
#include "gl_utils.hpp"

#include <array>

#include <glm/gtc/type_ptr.hpp>

#include "images.hpp"

namespace gl_utils {

template <typename Func>
void WithUniform(const GLuint handle, const std::string_view name, Func&& func) {
  glUseProgram(handle);
  const GLint uniform = glGetUniformLocation(handle, name.data());
  ASSERT(uniform != -1, "Failed to find uniform: {}", name);
  std::invoke(std::forward<Func>(func), uniform);
  glUseProgram(0);
}

void ShaderProgram::SetMatrixUniform(const std::string_view name, const glm::mat4x4& matrix) const {
  WithUniform(Handle(), name, [&](GLint uniform) { glUniformMatrix4fv(uniform, 1, GL_FALSE, glm::value_ptr(matrix)); });
}

void ShaderProgram::SetMatrixUniform(std::string_view name, const glm::mat3x3& matrix) const {
  WithUniform(Handle(), name, [&](GLint uniform) { glUniformMatrix3fv(uniform, 1, GL_FALSE, glm::value_ptr(matrix)); });
}

void ShaderProgram::SetUniformVec2(const std::string_view name, const glm::vec2 value) const {
  WithUniform(Handle(), name, [&](GLint uniform) { glUniform2f(uniform, value.x, value.y); });
}

void ShaderProgram::SetUniformInt(const std::string_view name, const GLint value) const {
  WithUniform(Handle(), name, [&](GLint uniform) { glUniform1i(uniform, value); });
}

// TODO: Fail more gracefully maybe?
ShaderProgram CompileShaderProgram(const std::string_view vertex_source, const std::string_view fragment_source) {
  const Shader vertex_shader{GL_VERTEX_SHADER};
  ASSERT(vertex_shader, "Failed to allocate vertex shader");

  // Create array of strings, which is what glShaderSource expects (we only have one string).
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

// Convert channels + depth to our preferred representation on the GPU.
static GLenum GetTextureRepresentation(const int channels, const images::ImageDepth depth) {
  using images::ImageDepth;
  constexpr std::array table = {
      TextureFormatEntry(1, ImageDepth::Bits8, GL_R8), TextureFormatEntry(3, ImageDepth::Bits8, GL_RGB8),
      TextureFormatEntry(1, ImageDepth::Bits16, GL_R16), TextureFormatEntry(3, ImageDepth::Bits32, GL_RGB32F)};

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

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, 0);
}

TextureCube::TextureCube() : OpenGLHandle(CreateTexture(), [](GLuint x) noexcept { glDeleteTextures(1, &x); }) {}

// Get appropriate OpenGL for the given face index.
static GLenum TargetForFace(int face) {
  constexpr std::array<GLenum, 6> faces = {
      GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X, GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
      GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
  };
  ASSERT(face >= 0 && face < 6, "Invalid face: {}", face);
  return faces[face];
}

void TextureCube::Fill(const int face, const images::SimpleImage& image) {
  ASSERT(image.width == image.height, "Faces should be square. Width = {}, height = {}", image.width, image.height);

  glBindTexture(GL_TEXTURE_CUBE_MAP, Handle());
  if (dimension_ == 0) {
    // Allocate cubemap:
    dimension_ = image.width;
    glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GetTextureRepresentation(image.components, image.depth), dimension_,
                   dimension_);
  } else {
    ASSERT(dimension_ == image.width, "All faces must have same dimension");
  }

  // Copy face to GPU:
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
  glTexSubImage2D(TargetForFace(face), 0, 0, 0, dimension_, dimension_, GetTextureInputFormat(image.components),
                  GetTextureDataType(image.depth), &image.data[0]);

  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#if 0
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
#endif
}

inline GLuint CreateVertexArray() {
  GLuint array{0};
  glGenVertexArrays(1, &array);
  return array;
}

inline GLuint CreateBuffer() {
  GLuint buffer{0};
  glGenBuffers(1, &buffer);
  return buffer;
}

FullScreenQuad::FullScreenQuad()
    : vertex_array_(CreateVertexArray(), [](GLuint x) noexcept { glDeleteVertexArrays(1, &x); }),
      vertex_buffer_(CreateBuffer(), [](GLuint x) noexcept { glDeleteBuffers(1, &x); }),
      index_buffer_(CreateBuffer(), [](GLuint x) noexcept { glDeleteBuffers(1, &x); }) {
  // Points for the FullScreenQuad:
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

  // Send vertex data.
  glBindVertexArray(vertex_array_.Handle());
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_.Handle());
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

  // Send triangle data
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_.Handle());
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * triangles.size(), &triangles[0], GL_STATIC_DRAW);

  // Specify how vertices are arranged in the buffer:
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
  glEnableVertexAttribArray(0);

  // Specify how texture coordinates are arranged in the buffer:
  // The last argument is just an offset, encoded as void*.
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // Unbind
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

void FullScreenQuad::Draw(const ShaderProgram& program) const {
  ASSERT(program, "Program is not initialized");
  glUseProgram(program.Handle());
  glBindVertexArray(vertex_array_.Handle());
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
  glUseProgram(0);
  glBindVertexArray(0);
}

inline GLuint CreateFramebuffer() {
  GLuint fbo{0};
  glGenFramebuffers(1, &fbo);
  ASSERT(fbo, "Failed to create FBO");
  return fbo;
}

FramebufferObject::FramebufferObject(const int width, const int height)
    : fbo_(CreateFramebuffer(), [](GLuint x) noexcept { glDeleteFramebuffers(1, &x); }),
      texture_(CreateTexture(), [](GLuint x) noexcept { glDeleteTextures(1, &x); }),
      width_(width),
      height_(height) {
  ASSERT(width_ > 0 && height_ > 0);

  glBindFramebuffer(GL_FRAMEBUFFER, fbo_.Handle());
  glBindTexture(GL_TEXTURE_2D, texture_.Handle());

  // Allocate storage.
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, width_, height_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  // Attach texture to the FBO.
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_.Handle(), 0);
  const GLenum fbo_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  ASSERT(fbo_status == GL_FRAMEBUFFER_COMPLETE, "FBO is not complete, status = {}", fbo_status);

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glBindTexture(GL_TEXTURE_2D, 0);
}

images::SimpleImage FramebufferObject::ReadContents(const int channels, const images::ImageDepth depth) const {
  images::SimpleImage output{width_, height_, channels, depth};
  // Bind and read the texture back:
  glBindTexture(GL_TEXTURE_2D, texture_.Handle());
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glGetTexImage(GL_TEXTURE_2D, 0, GetTextureInputFormat(channels), GetTextureDataType(depth), &output.data[0]);
  glBindTexture(GL_TEXTURE_2D, 0);
  return output;
}

void FramebufferObject::ReadIntoPixelbuffer(const int channels, const images::ImageDepth depth,
                                            const GLuint buffer_handle) const {
  glBindBuffer(GL_PIXEL_PACK_BUFFER, buffer_handle);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo_.Handle());
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  // Pass null (the pbo is bound, and data will go there).
  glReadPixels(0, 0, width_, height_, GetTextureInputFormat(channels), GetTextureDataType(depth), nullptr);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

inline GLuint CreatePixelBuffer() {
  GLuint result{0};
  glGenBuffers(1, &result);
  ASSERT(result != 0, "Failed to create pixel buffer");
  return result;
}

PixelbufferQueue::PixelbufferQueue(std::size_t num_buffers, const int width, const int height, const int channels,
                                   const images::ImageDepth depth)
    : channels_(channels), depth_(depth), width_(width), height_(height) {
  ASSERT(num_buffers > 0);
  pbo_.reserve(num_buffers);
  pending_reads_.reserve(num_buffers);
  for (std::size_t i = 0; i < num_buffers; ++i) {
    pbo_.emplace_back(CreatePixelBuffer(), [](GLuint x) noexcept { glDeleteBuffers(1, &x); });
  }
  // Allocate the buffers:
  for (const OpenGLHandle& buffer : pbo_) {
    glBindBuffer(GL_PIXEL_PACK_BUFFER, buffer.Handle());
    glBufferData(GL_PIXEL_PACK_BUFFER, width * height * channels * static_cast<int>(depth), nullptr, GL_STREAM_READ);
  }
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

std::optional<std::pair<PixelbufferQueue::Key, images::SimpleImage>> PixelbufferQueue::QueueReadFromFbo(
    const Key new_key, const FramebufferObject& fbo) {
  std::optional<std::pair<PixelbufferQueue::Key, images::SimpleImage>> result{};
  if (pending_reads_.size() < pbo_.size()) {
    // The queue is not full, so put this in the next available slot.
    const std::size_t read_index = pending_reads_.size();
    fbo.ReadIntoPixelbuffer(channels_, depth_, pbo_[read_index].Handle());
    pending_reads_.emplace_back(read_index, new_key);
  } else {
    // We need to pop a read from the front of the queue
    const auto [read_index, returned_key] = pending_reads_.front();
    pending_reads_.erase(pending_reads_.begin());
    ASSERT(read_index < pbo_.size());

    // Bind it fetch the data:
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_[read_index].Handle());
    const void* mapped = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
    ASSERT(mapped, "Failed to map PBO");

    // Copy it out to an image and place it in `result` so we return something:
    images::SimpleImage output_image{width_, height_, channels_, depth_};
    std::memcpy(&output_image.data[0], mapped, output_image.data.size());
    result.emplace(returned_key, std::move(output_image));

    // Unmap the buffer.
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    // Queue another read w/ the new key:
    fbo.ReadIntoPixelbuffer(channels_, depth_, pbo_[read_index].Handle());
    pending_reads_.emplace_back(read_index, new_key);
  }
  return result;
}

void GLAPIENTRY MessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei, const GLchar* message,
                                const void*) {
  fmt::print("GL error callback: source = {:X}, type = {:X}, id = {}, severity = {:X}, message = \'{}\'\n", source,
             type, id, severity, message);
}

void EnableDebugOutput(int glad_version) {
  // During init, enable debug output
  if (GLAD_VERSION_MAJOR(glad_version) >= 4 && GLAD_VERSION_MINOR(glad_version) >= 3) {
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(MessageCallback, nullptr);
  } else {
    fmt::print("GL debug log not supported.\n");
  }
}

}  // namespace gl_utils

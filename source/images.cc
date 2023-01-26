// Copyright 2023 Gareth Cross
#include "images.hpp"

#include <array>
#include <execution>
#include <memory>

#include <fmt/format.h>
#include <png.h>
#include <scope_guard.hpp>

// Use STB for reading PNGs, as it is much simpler.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_FAILURE_STRINGS
#define STBI_WINDOWS_UTF8
#include <stb_image.h>

#include "assertions.hpp"

namespace images {

std::optional<SimpleImage> LoadPng(const std::filesystem::path& path, const ImageDepth expected_depth) {
  const std::string path_str = path.u8string();
  ASSERT(expected_depth != ImageDepth::Bits32, "Cannot load 32 bit images w/ stb.");

  // Delegate to either 16bit or 8bit implementation:
  SimpleImage image{};
  image.depth = expected_depth;
  if (expected_depth == ImageDepth::Bits16) {
    std::unique_ptr<uint16_t[], void (*)(void*)> data{nullptr, &stbi_image_free};
    data.reset(stbi_load_16(path_str.c_str(), &image.width, &image.height, &image.components, 0));
    if (!data) {
      return {};
    }
    // Copy:
    image.Allocate();
    std::memcpy(&image.data[0], data.get(), image.data.size());
  } else {
    std::unique_ptr<uint8_t[], void (*)(void*)> data{nullptr, &stbi_image_free};
    data.reset(stbi_load(path_str.c_str(), &image.width, &image.height, &image.components, 0));
    if (!data) {
      return {};
    }
    image.Allocate();
    std::memcpy(&image.data[0], data.get(), image.data.size());
  }

  return {std::move(image)};
}

void ErrorFunc(png_structp const ctx, png_const_charp const message) {
  const auto* path = static_cast<const std::string*>(png_get_error_ptr(ctx));
  fmt::print("Error while writing PNG file [{}]. Message: {}", *path, message);
}

void WarnFunc(png_structp const ctx, png_const_charp const message) {
  const auto* path = static_cast<const std::string*>(png_get_error_ptr(ctx));
  fmt::print("Warning while writing PNG file [{}]. Message: {}", *path, message);
}

void WriteFunc(png_structp const ctx, png_bytep const data, const png_size_t length) {
  ASSERT(ctx);
  std::ofstream* const output_stream = static_cast<std::ofstream*>(png_get_io_ptr(ctx));
  output_stream->write(reinterpret_cast<const char*>(data), static_cast<std::size_t>(length));
}

void FlushFunc(png_structp const ctx) {
  ASSERT(ctx);
  std::ofstream* const output_stream = static_cast<std::ofstream*>(png_get_io_ptr(ctx));
  output_stream->flush();
}

// We have to use libpng for this, since STB cannot write 16-bit pngs.
void WritePng(const std::filesystem::path& path, const SimpleImage& image, const bool flip_vertical) {
  ASSERT(!image.data.empty());
  ASSERT(image.components == 1 || image.components == 3, "Invalid # of components: {}", image.components);
  ASSERT(image.data.size() == image.Stride() * image.height, "Invalid image dims. size = {}, stride = {}, height = {}",
         image.data.size(), image.Stride(), image.height);
  ASSERT(image.depth == ImageDepth::Bits8 || image.depth == ImageDepth::Bits16, "Invalid bit depth for WritePng: {}",
         static_cast<int>(image.depth));

  const std::string path_str = path.u8string();
  png_struct* png_writer = png_create_write_struct(
      PNG_LIBPNG_VER_STRING, const_cast<void*>(static_cast<const void*>(&path_str)), &ErrorFunc, &WarnFunc);
  ASSERT(png_writer, "Failed to create PNG writer while writing [{}]", path_str);

  png_set_compression_level(png_writer, 6);

  png_info* info = png_create_info_struct(png_writer);
  const auto cleanup = sg::make_scope_guard([&]() { png_destroy_write_struct(&png_writer, &info); });
  ASSERT(info, "Failed to create info struct while writing [{}]", path_str);

  // Open file handle:
  std::ofstream output_file{path, std::ios::out | std::ios::binary};
  ASSERT(output_file.good(), "Failed to open output file: {}", path_str);

  // Configure to write to the file stream:
  png_set_write_fn(png_writer, static_cast<void*>(&output_file), &WriteFunc, &FlushFunc);

  // Setup + write header
  png_set_IHDR(png_writer, info, static_cast<png_uint_32>(image.width), static_cast<png_uint_32>(image.height),
               static_cast<int>(image.depth) * 8, image.components == 3 ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_GRAY,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png_writer, info);

  // Create array of row pointers and write out the image:
  std::vector<png_bytep> row_pointers{static_cast<std::size_t>(image.height)};
  for (std::size_t y = 0; y < image.height; ++y) {
    if (flip_vertical) {
      // Flip the order of rows when we write out:
      row_pointers[y] = const_cast<uint8_t*>(&image.data[(image.height - y - 1) * image.Stride()]);
    } else {
      row_pointers[y] = const_cast<uint8_t*>(&image.data[y * image.Stride()]);
    }
  }
  png_write_image(png_writer, &row_pointers[0]);
  png_write_end(png_writer, info);
  output_file.flush();
}

SimpleImage LoadRawFloatImage(const std::filesystem::path& path, const int width, const int height,
                              const int channels) {
  std::ifstream stream(path, std::ios::in | std::ios::binary);
  ASSERT(stream.good(), "Failed to open file: {}", path.u8string());

  // get file size:
  stream.unsetf(std::ios::skipws);
  stream.seekg(0, std::ios::end);
  const std::streampos file_size = stream.tellg();
  stream.seekg(0, std::ios::beg);

  const auto expected_size = static_cast<std::size_t>(width * height * channels * sizeof(float));
  ASSERT(file_size == expected_size,
         "File is the wrong size. Expected = width ({}) * height ({}) * channels ({}) * {} = {}, actual = {}", width,
         height, channels, sizeof(float), expected_size, file_size);

  // Read the entire file:
  SimpleImage image(width, height, channels, ImageDepth::Bits32);
  const auto last =
      std::copy(std::istream_iterator<uint8_t>(stream), std::istream_iterator<uint8_t>(), image.data.begin());
  ASSERT(last == image.data.end(), "Failed to read the entire file for some reason. File position: {}", stream.tellg());
  return image;
}

std::vector<std::optional<SimpleImage>> LoadCubemapImages(const std::filesystem::path& dataset_root,
                                                          const std::size_t image_index, const std::size_t camera_index,
                                                          const CubemapType type) {
  constexpr std::array<std::size_t, 6> faces = {0, 1, 2, 3, 4, 5};
  const std::string_view sub_folder = (type == CubemapType::Rgb) ? "image" : "depth";

  // load pngs in parallel:
  std::vector<std::optional<SimpleImage>> images_out{faces.size()};
  std::transform(std::execution::parallel_unsequenced_policy{}, faces.begin(), faces.end(), images_out.begin(),
                 [&](const std::size_t face_index) {
                   // Path to the specific sub-image
                   const std::filesystem::path path = dataset_root / sub_folder /
                                                      fmt::format("camera{:02}", camera_index) /
                                                      fmt::format("{:08}_{:02}.png", image_index, face_index);
                   // 8-bit for color, 16-bit for inverse depth:
                   return images::LoadPng(path, type == CubemapType::Rgb ? ImageDepth::Bits8 : ImageDepth::Bits16);
                 });
  return images_out;
}

}  // namespace images

// Copyright 2023 Gareth Cross
#pragma once
#include <filesystem>
#include <optional>

namespace images {

// Supported bit depths.
enum class ImageDepth {
  Bits8 = 1,
  Bits16 = 2,
  Bits32 = 4,  //  Assumed to mean float.
};

// Very simple image type.
struct SimpleImage {
  std::vector<uint8_t> data{};
  int width{0};
  int height{0};
  int components{0};
  ImageDepth depth{ImageDepth::Bits8};

  SimpleImage() = default;

  // Construct w/ fields initialized.
  SimpleImage(int width, int height, int components, ImageDepth depth)
      : width(width), height(height), components(components), depth(depth) {
    Allocate();
  }

  // Is the image empty.
  [[nodiscard]] bool IsEmpty() const { return data.empty(); }

  // Length of a row in bytes.
  [[nodiscard]] std::size_t Stride() const {
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(depth) * static_cast<std::size_t>(components);
  }

  /// Allocate data to fit.
  void Allocate() { data.resize(Stride() * static_cast<std::size_t>(height)); }
};

// Load a PNG image.
SimpleImage LoadPng(const std::filesystem::path& path, ImageDepth expected_depth);

// Write a PNG image.
void WritePng(const std::filesystem::path& path, const SimpleImage& image, bool flip_vertical);

// Load a float image from a raw file (no header, just packed bytes).
// Data is expected to be in row-major order.
SimpleImage LoadRawFloatImage(const std::filesystem::path& path, int width, int height, int channels);

// Types of cubemaps:
enum class CubemapType {
  Rgb,
  Depth,
};

// Load all the cubemap images of a given type for the specified index.
std::vector<SimpleImage> LoadCubemapImages(const std::filesystem::path& dataset_root, std::size_t image_index,
                                           std::size_t camera_index, bool parallelize = false);

}  // namespace images

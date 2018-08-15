#pragma once
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include "fisheryatesshuffle.h"

enum class PreprocessingType : unsigned {
  NO_PREPROCESSING = 0,
  DEMEANED,
  NORMALIZED,
  STANDARDIZED
};

/**
 * @brief The Image class represents a multiple channels image
 * with intensities range [0, 255]
 */
class Image {
 public:
  enum class Channel : unsigned { RED = 0, GREEN, BLUE };

  Image() = default;
  Image(const size_t width, const size_t height, const size_t channels)
      : width_(width), height_(height), channels_(channels) {}
  Image(const std::vector<float>& float_image, const size_t width, const size_t height,
        const size_t channels)
      : Image(width, height, channels) {
    // Convert float image to the specified type
    // Supposed to be unsigned char for the most use cases
    data_.assign(float_image.begin(), float_image.end());
  }

  unsigned char operator()(const size_t x, const size_t y, const size_t c) const {
    return data_[x + y * width_ + c * width_ * height_];
  }
  unsigned char operator()(const size_t x, const size_t y, const Channel c) const {
    return operator()(x, y, static_cast<size_t>(c));
  }

  size_t width() const { return width_; }
  size_t height() const { return height_; }
  size_t channels() const { return channels_; }

 private:
  size_t width_, height_;
  size_t channels_;
  std::vector<unsigned char> data_;
};

class CIFAR10Reader {
 public:
  CIFAR10Reader() : state_(PreprocessingType::NO_PREPROCESSING) {}
  bool readBin(std::string filepath, const bool bias_trick);

  // Data processing
  void computeMeanImage();
  void computeStdImage();
  bool normalize();
  bool standardize();
  bool demean();
  bool reset();

  void setMeanImage(const std::vector<float>& mean_image) { mean_image_ = mean_image; }
  void setStdImage(const std::vector<float>& std_image) { std_image_ = std_image; }
  Image getImage(const size_t index) const;

  bool meanImageIsComputed() const;
  bool stdImageIsComputed() const;
  bool datasetWasPreprocessed() const;
  bool datasetIsLoaded() const;

  std::pair<float, float> minmax();

  std::vector<size_t> getBatchIdxs(int batch_size) const;

  const std::vector<float>& stdImage() const { return std_image_; }
  const std::vector<float>& meanImage() const { return mean_image_; }
  const std::vector<int>& labels() const { return labels_; }
  const std::vector<std::vector<float>>& images() const { return images_; }

 private:
  std::vector<int> labels_;
  std::vector<std::vector<float>> images_;
  std::vector<float> mean_image_;
  std::vector<float> std_image_;
  PreprocessingType state_;
  mutable FisherYatesShuffle shuffler;
  std::vector<std::vector<float>> const_images_;
};

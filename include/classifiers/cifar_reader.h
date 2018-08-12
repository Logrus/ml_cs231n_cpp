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

class CIFAR10Reader {
 public:
  CIFAR10Reader() : state_(PreprocessingType::NO_PREPROCESSING) {}
  bool read_bin(std::string filepath, bool bias_trick);

  // Data processing
  void compute_mean();
  void compute_std();
  void normalize();
  void standardize();
  void demean();
  void reset();

  void setMeanImage(const std::vector<float>& mean_image) { mean_image_ = mean_image; }
  void setStdImage(const std::vector<float>& std_image) { std_image_ = std_image; }

  std::pair<float, float> minmax();

  std::vector<size_t> get_batch_idxs(int batch_size) const;

  const std::vector<float>& std_image() const { return std_image_; }
  const std::vector<float>& mean_image() const { return mean_image_; }
  const std::vector<int>& labels() const { return labels_; }
  const std::vector<std::vector<float>>& images() const { return images_; }

 private:
  std::vector<int> labels_;
  std::vector<std::vector<float>> images_;
  std::vector<float> mean_image_;
  std::vector<float> std_image_;
  PreprocessingType state_;
  mutable FisherYatesShuffle shuffler;
  std::vector<std::vector<float>> images_copy_;
};

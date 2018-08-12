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

  std::vector<int> labels_;
  std::vector<std::vector<float> > images_;

  // Data processing
  void compute_mean();
  void compute_std();
  void normalize();
  void standardize();
  void demean();
  void reset();
  std::vector<float> mean_image;
  std::vector<float> std_image;

  std::pair<float, float> minmax();

  std::vector<size_t> get_batch_idxs(int batch_size) const;

 private:
  PreprocessingType state_;
  mutable FisherYatesShuffle shuffler;
  std::vector<std::vector<float> > images_copy_;
};

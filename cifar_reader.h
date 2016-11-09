#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <tuple>
#include <set>
#include "fisheryatesshuffle.h"

class CIFAR10Reader {
public:
  bool read_bin(std::string filepath, bool bias_trick);
  
  std::vector<int> labels_;
  std::vector< std::vector<float> > images_;

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

  std::vector<int> get_batch_idxs(int batch_size);

private:
  FisherYatesShuffle shuffler;
  std::vector< std::vector<float> > images_copy_;
};


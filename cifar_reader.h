#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

class CIFAR10Reader {
public:
  bool read_bin(std::string filepath);
  
  std::vector<int> labels_;
  std::vector< std::vector<float> > images_;

  // Data processing
  void normalize();
  void standardize();
  // mean_image

private:
  std::vector< std::vector<float> > images_copy_;
};


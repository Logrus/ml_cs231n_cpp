#pragma once
#include "classifier.h"

#include <uni_freiburg_cv/CMatrix.h>

#include <assert.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

class LinearSoftmax : public Classifier {
 public:
  LinearSoftmax(const unsigned, const unsigned);

  /**
   * @brief Computes SVM L2 regularization term W^2
   * @return float representing elementwise sum of squares of weight matrix
   * elements
   */
  float weightRegularizationL2() const override;

  float lossForSingleImage(const std::vector<float>& image, const int& y);
  float computeLoss(const std::vector<std::vector<float> >& images,
                    const std::vector<int>& labels,
                    const std::vector<size_t>& indexies) override;

  size_t infer(const std::vector<float>& image) const override;

  std::vector<float> inferenceLoss(const std::vector<float>& image,
                                   const size_t y) const override;
};

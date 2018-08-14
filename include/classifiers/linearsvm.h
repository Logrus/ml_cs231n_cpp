#ifndef LINEARSVM_H
#define LINEARSVM_H
#include <assert.h>
#include <uni_freiburg_cv/CMatrix.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
#include "classifier.h"

class LinearSVM : public Classifier {
 public:
  LinearSVM(const unsigned, const unsigned);

  /**
   * @brief Computes SVM L2 regularization term W^2
   * @return float representing elementwise sum of squares of weight matrix elements
   */
  float weightRegularizationL2() const override;

  float lossForSingleImage(const std::vector<float>& image, const int& y) override;
  float computeLoss(const std::vector<std::vector<float> >& images, const std::vector<int>& labels,
                    const std::vector<size_t>& indexies) override;

  int infer(const std::vector<float>& image) const override;

  std::vector<float> inferenceLoss(const std::vector<float>& image, const size_t y) const override;
};

#endif  // LINEARSVM_H

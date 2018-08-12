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
  LinearSVM(int, int);

  /**
   * @brief Computes SVM L2 regularization term W^2
   * @return float representing elementwise sum of squares of weight matrix elements
   */
  float L2W_reg();

  float loss_one_image(const std::vector<float>& image, const int& y);
  float loss(const std::vector<std::vector<float> >& images, const std::vector<int>& labels,
             const std::vector<size_t>& indexies) override;

  int inference(const std::vector<float>& image);

  std::vector<float> inference_loss(const std::vector<float>& image, const int& y);
};

#endif  // LINEARSVM_H

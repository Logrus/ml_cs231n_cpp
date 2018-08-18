#pragma once
#include <uni_freiburg_cv/CMatrix.h>
#include <chrono>
#include <random>

class ClassifierInterface {
 public:
  virtual float weightRegularizationL2() const = 0;
  virtual float lossForSingleImage(const std::vector<float>& image, const int& y) = 0;
  virtual float computeLoss(const std::vector<std::vector<float> >& images,
                            const std::vector<int>& labels,
                            const std::vector<size_t>& indexies) = 0;
  virtual size_t infer(const std::vector<float>& image) const = 0;
  virtual std::vector<float> inferenceLoss(const std::vector<float>& image,
                                           const size_t y) const = 0;
  virtual ~ClassifierInterface();
};

class Classifier : public ClassifierInterface {
 public:
  /**
   * @brief Classifier
   * @param dimentionality_classes
   * @param dimentionality_data
   */
  Classifier(const unsigned dimentionality_classes, const unsigned dimentionality_data);

  /**
   * @brief initializeW Initializes weights with samples form zero mean normal distribution
   */
  virtual void initializeW(const float std = 0.00001f);

  virtual void copyW(const CMatrix<float>& W_);

  /**
   * @brief Perform computation of scores in a loss function
   * which is typically score = W*x
   * where W are internally kept weights, randomly initialized at the beginning
   * W [dimentionality_classes x dimentionality_data]
   * @param x [dimentionality_data x 1]
   * @return vector with scores [dimentionality_classes x 1]'
   */
  virtual std::vector<float> computeScores(const std::vector<float>& x) const;

  /**
   * @brief Computation of the weight to update ratio
   * the ratio of the update magnitudes to the value magnitudes
   * Link: http://cs231n.github.io/neural-networks-3/#ratio
   * @return weight_ratio
   */
  float computeWeightRatio() const;

  /// \todo protected:
  CMatrix<float> W_;   /// Weights of the loss
  CMatrix<float> dW_;  /// Gradients of weights

  unsigned classes_dim_;  /// Number of classes
  unsigned data_dim_;     /// Input data dimentionality

  double lambda_;         /// Regularization
  double learning_rate_;  /// Learning rate
};

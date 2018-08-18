#include <classifiers/linearsvm.h>

LinearSVM::LinearSVM(const unsigned classes, const unsigned dimentionality)
    : Classifier(classes, dimentionality) {}

float LinearSVM::weightRegularizationL2() const {
  float sum = 0;
  for (int x = 0; x < W_.xSize(); ++x)
    for (int y = 0; y < W_.ySize(); ++y) {
      sum += W_(x, y) * W_(x, y);
    }
  return sum;
}

float LinearSVM::lossForSingleImage(const std::vector<float>& image, const int& y) {
  // Compute scores
  // scores = W*x
  std::vector<float> scores = computeScores(image);

  // Compute loss
  float loss = 0;
  std::vector<float> margins(classes_dim_, 0);
  for (size_t j = 0; j < classes_dim_; ++j) {
    if (j == y) continue;
    float margin = scores[j] - scores[y] + 1;

    loss += std::max(0.f, margin);

    // Compute gradient
    if (margin > 0) {
      for (size_t d = 0; d < data_dim_; ++d) {
        dW_(y, d) -= image[d];
        dW_(j, d) += image[d];
      }
    }
  }

  return loss;
}

float LinearSVM::computeLoss(const std::vector<std::vector<float> >& images,
                             const std::vector<int>& labels, const std::vector<size_t>& indexies) {
  // Reset gradient
  dW_.fill(0.0);

  // Compute loss for all images
  float L = 0;
  const size_t N = indexies.size();  // N images in batch
  for (size_t i = 0; i < N; ++i) {
    L += lossForSingleImage(images[indexies[i]], labels[indexies[i]]);
  }
  L /= static_cast<float>(N);
  L += 0.5 * lambda_ * weightRegularizationL2();

  // Normalize and regularize gradient
  for (int x = 0; x < dW_.xSize(); ++x) {
    for (int y = 0; y < dW_.ySize(); ++y) {
      dW_(x, y) = dW_(x, y) / static_cast<float>(N) + lambda_ * W_(x, y);
    }
  }

  // Update weights
  for (int x = 0; x < W_.xSize(); ++x) {
    for (int y = 0; y < W_.ySize(); ++y) {
      W_(x, y) -= learning_rate_ * dW_(x, y);
    }
  }

  return L;
}

size_t LinearSVM::infer(const std::vector<float>& image) const {
  // scores = W*x
  std::vector<float> scores = computeScores(image);
  // Get the index of the max element
  return std::max_element(scores.begin(), scores.end()) - scores.begin();
}

std::vector<float> LinearSVM::inferenceLoss(const std::vector<float>& image, const size_t y) const {
  // Compute scores
  // scores = W*x
  std::vector<float> scores = computeScores(image);

  // Compute loss
  std::vector<float> margins(classes_dim_, 0);
  for (size_t j = 0; j < classes_dim_; ++j) {
    if (j == y) continue;
    margins[j] = std::max(0.f, scores[j] - scores[y] + 1);
  }

  return margins;
}

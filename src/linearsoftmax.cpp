#include <classifiers/linearsoftmax.h>

LinearSoftmax::LinearSoftmax(const unsigned classes, const unsigned dimentionality)
    : Classifier(classes, dimentionality) {}

float LinearSoftmax::weightRegularizationL2() const {
  float sum = 0;
  for (int x = 0; x < W_.xSize(); ++x)
    for (int y = 0; y < W_.ySize(); ++y) {
      sum += W_(x, y) * W_(x, y);
    }
  return sum;
}

float LinearSoftmax::lossForSingleImage(const std::vector<float>& image, const int& y) {
  // Compute scores
  // scores = W*x
  std::vector<float> scores = computeScores(image);

  // Compute loss

  // 1. Normalize scores for numerical stability
  // Find max and subtract it from every score
  const float max = *(std::max_element(scores.begin(), scores.end()));
  for (auto& a : scores) {
    a -= max;
  }

  // 2. Scores are unnormalized log probabilities
  // we need to exp to get unnormalized probabilities
  std::vector<double> unnormalized_prob(classes_dim_, 0);
  for (size_t j = 0; j < classes_dim_; ++j) {
    unnormalized_prob[j] = std::exp(scores[j]);
  }

  // 3. Normalize to get probabilities
  const float normalizer =
      std::accumulate(unnormalized_prob.begin(), unnormalized_prob.end(), 0.0f);
  std::vector<double> probabilities(classes_dim_, 0);
  for (size_t j = 0; j < classes_dim_; ++j) {
    probabilities[j] = unnormalized_prob[j] / normalizer;
  }

  // 4. Compute loss
  const float loss = -std::log(probabilities[y]);

  // Compute gradient
  // Take derivative for scores
  std::vector<double> dscores(probabilities);
  dscores[y] -= 1;
  // Propagate it to weights
  // dW = x*dscores
  for (size_t c = 0; c < classes_dim_; ++c) {
    for (size_t d = 0; d < data_dim_; ++d) {
      dW_(c, d) += image[d] * dscores[c];
    }
  }

  return loss;
}

float LinearSoftmax::computeLoss(const std::vector<std::vector<float> >& images,
                                 const std::vector<int>& labels,
                                 const std::vector<size_t>& indexies) {
  // Reset gradient
  dW_.fill(0.0);

  // Compute loss for all images
  float L = 0;
  size_t N = indexies.size();  // N images in batch
  for (size_t i = 0; i < N; ++i) {
    L += lossForSingleImage(images[indexies[i]], labels[indexies[i]]);
  }
  L /= static_cast<float>(N);
  L += 0.5f * lambda_ * weightRegularizationL2();

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

int LinearSoftmax::infer(const std::vector<float>& image) const {
  const std::vector<float> scores = computeScores(image);
  // Get the index of the max element
  return std::max_element(scores.begin(), scores.end()) - scores.begin();
}

std::vector<float> LinearSoftmax::inferenceLoss(const std::vector<float>& image,
                                                const size_t y) const {
  // Compute scores
  // scores = W*x
  std::vector<float> scores = computeScores(image);

  // 1. Normalize scores for numerical stability
  // Find max and subtract it from every score
  const float max = *(std::max_element(scores.begin(), scores.end()));
  for (auto& a : scores) {
    a -= max;
  }

  // 2. Scores are unnormalized log probabilities
  // we need to exp to get unnormalized probabilities
  std::vector<float> unnormalized_probabilities(classes_dim_, 0);
  for (size_t j = 0; j < classes_dim_; ++j) {
    unnormalized_probabilities[j] = std::exp(scores[j]);
  }

  // 3. Normalize to get probabilities
  const float normalizer =
      std::accumulate(unnormalized_probabilities.begin(), unnormalized_probabilities.end(), 0.0f);
  std::vector<float> probabilities(classes_dim_, 0);
  for (size_t j = 0; j < classes_dim_; ++j) {
    probabilities[j] = unnormalized_probabilities[j] / normalizer;
  }

  return probabilities;
}

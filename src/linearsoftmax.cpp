#include <classifiers/linearsoftmax.h>

LinearSoftmax::LinearSoftmax(int classes, int dimentionality)
    : Classifier(classes, dimentionality) {}

float LinearSoftmax::L2W_reg() {
  float sum = 0;
  for (int x = 0; x < W.xSize(); ++x)
    for (int y = 0; y < W.ySize(); ++y) {
      sum += W(x, y) * W(x, y);
    }
  return sum;
}

float LinearSoftmax::loss_one_image(const std::vector<float>& image, const int& y) {
  assert(image.size() == 3073);

  std::vector<float> scores(10, 0);

  // Compute scores
  // scores = W*x
  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < D; ++d) {
      scores[c] += W(c, d) * image[d];
    }
  }

  // Compute loss

  // 1. Normalize scores for numerical stability
  // Find max and subtract it from every score
  float max = *(std::max_element(scores.begin(), scores.end()));
  for (auto& a : scores) {
    a -= max;
  }

  // 2. Scores are unnormalized log probabilities
  // we need to exp to get unnormalized probabilities
  std::vector<double> unnormalized_prob(10, 0);
  for (int j = 0; j < C; ++j) {
    unnormalized_prob[j] = std::exp(scores[j]);
  }

  // 3. Normalize to get probabilities
  float normalizer = std::accumulate(unnormalized_prob.begin(), unnormalized_prob.end(), 0.0f);
  std::vector<double> prob(10, 0);
  for (int j = 0; j < C; ++j) {
    prob[j] = unnormalized_prob[j] / normalizer;
  }

  // 4. Compute loss
  float loss = -std::log(prob[y]);

  // Compute gradient
  // Take derivative for scores
  std::vector<double> dscores(prob);
  dscores[y] -= 1;
  // Propagate it to weights
  // dW = x*dscores
  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < D; ++d) {
      dW(c, d) += image[d] * dscores[c];
    }
  }

  return loss;
}

float LinearSoftmax::loss(const std::vector<std::vector<float> >& images,
                          const std::vector<int>& labels, const std::vector<size_t>& indexies) {
  assert(images.size() == 50000);
  assert(C == 10);
  assert(D == 3073);

  // Reset gradient
  dW.fill(0.0);

  // Compute loss for all images
  float L = 0;
  size_t N = indexies.size();  // N images in batch
  for (size_t i = 0; i < N; ++i) {
    L += loss_one_image(images[indexies[i]], labels[indexies[i]]);
  }
  L /= static_cast<float>(N);
  L += 0.5f * lambda * L2W_reg();

  // Normalize and regularize gradient
  for (int x = 0; x < dW.xSize(); ++x) {
    for (int y = 0; y < dW.ySize(); ++y) {
      dW(x, y) = dW(x, y) / static_cast<float>(N) + lambda * W(x, y);
    }
  }

  // Update weights
  for (int x = 0; x < W.xSize(); ++x) {
    for (int y = 0; y < W.ySize(); ++y) {
      W(x, y) -= learning_rate * dW(x, y);
    }
  }

  return L;
}

int LinearSoftmax::inference(const std::vector<float>& image) {
  std::vector<float> scores(10, 0);

  // scores = W*x
  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < D; ++d) {
      scores[c] += W(c, d) * image[d];
    }
  }

  return std::max_element(scores.begin(), scores.end()) - scores.begin();
}

std::vector<float> LinearSoftmax::inference_loss(const std::vector<float>& image, const int& y) {
  std::vector<float> scores(10, 0);

  // Compute scores
  // scores = W*x
  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < D; ++d) {
      scores[c] += W(c, d) * image[d];
    }
  }

  // 1. Normalize scores for numerical stability
  // Find max and subtract it from every score
  float max = *(std::max_element(scores.begin(), scores.end()));
  for (auto& a : scores) {
    a -= max;
  }

  // 2. Scores are unnormalized log probabilities
  // we need to exp to get unnormalized probabilities
  std::vector<double> unnormalized_prob(10, 0);
  for (int j = 0; j < C; ++j) {
    unnormalized_prob[j] = std::exp(scores[j]);
  }

  // 3. Normalize to get probabilities
  float normalizer = std::accumulate(unnormalized_prob.begin(), unnormalized_prob.end(), 0.0f);
  std::vector<float> prob(10, 0);
  for (int j = 0; j < C; ++j) {
    prob[j] = unnormalized_prob[j] / normalizer;
  }

  return prob;
}

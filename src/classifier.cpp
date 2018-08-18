#include <classifiers/classifier.h>

// Has to be in cpp file to avoid
// -Wweak-vtables warning
ClassifierInterface::~ClassifierInterface() = default;

Classifier::Classifier(const unsigned dimentionality_classes,
                       const unsigned dimentionality_data)
    : classes_dim_(dimentionality_classes),
      data_dim_(dimentionality_data),
      lambda_(0.5),
      learning_rate_(1.0e-8) {
  W_.setSize(dimentionality_classes, dimentionality_data);
  dW_.setSize(dimentionality_classes, dimentionality_data);

  // Initialize gradient
  dW_.fill(0.0);
}

void Classifier::copyW(const CMatrix<float>& inW) { W_ = inW; }

std::vector<float> Classifier::computeScores(
    const std::vector<float>& x) const {
  std::vector<float> scores(classes_dim_, 0.f);

  for (size_t c = 0; c < classes_dim_; ++c) {
    for (size_t d = 0; d < data_dim_; ++d) {
      scores[c] += W_(c, d) * x[d];
    }
  }

  return scores;
}

float Classifier::computeWeightRatio() const {
  long double weight = 0, update = 0;
  for (int x = 0; x < W_.xSize(); ++x) {
    for (int y = 0; y < W_.ySize(); ++y) {
      weight += static_cast<long double>(W_(x, y) * W_(x, y));
      update += static_cast<long double>(dW_(x, y) * learning_rate_ *
                                         dW_(x, y) * learning_rate_);
    }
  }
  // Narrowing down precision after computing ratio
  return static_cast<float>(sqrt(update) / sqrt(weight));
}

void Classifier::initializeW(const float std) {
  long seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<float> distribution(0, std);
  for (int x = 0; x < W_.xSize(); ++x) {
    for (int y = 0; y < W_.ySize(); ++y) {
      W_(x, y) = distribution(generator);
      // make the weight for bias positive (better for initialization)
      if (y == W_.ySize() - 1) {
        W_(x, y) = fabs(W_(x, y));
      }
    }
  }
}

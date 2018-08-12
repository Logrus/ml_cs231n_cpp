#include <classifiers/classifier.h>

Classifier::Classifier(const int classes, const int dimentionality)
    : C(classes), D(dimentionality), lambda(0.5), learning_rate(1.0e-8f) {
  W.setSize(classes, dimentionality);
  dW.setSize(classes, dimentionality);

  // Initialize gradient
  dW.fill(0.0);
}

void Classifier::copyW(const CMatrix<float> inW) { W = inW; }

std::vector<float> Classifier::scores(const std::vector<float>& image) {
  std::vector<float> scores(10, 0);

  // scores = W*x
  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < D; ++d) {
      scores[c] += W(c, d) * image[d];
    }
  }

  return scores;
}

float Classifier::weight_ratio() {
  long double weight = 0, update = 0;
  for (int x = 0; x < W.xSize(); ++x) {
    for (int y = 0; y < W.ySize(); ++y) {
      weight += W(x, y) * W(x, y);
      update += dW(x, y) * learning_rate * dW(x, y) * learning_rate;
    }
  }
  return sqrt(update) / sqrt(weight);
}

void Classifier::initializeW() {
  // Randomly initialize weights
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<float> distribution(0, 0.00001);
  for (int x = 0; x < W.xSize(); ++x) {
    for (int y = 0; y < W.ySize(); ++y) {
      W(x, y) = distribution(generator);
      if (y == W.ySize() - 1) {  // make the weight for bias positive (better for initialization)
        W(x, y) = fabs(W(x, y));
      }
    }
  }
}

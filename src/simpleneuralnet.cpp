#include <classifiers/simpleneuralnet.h>

SimpleNeuralNet::SimpleNeuralNet(int input_size, int hidden_size,
                                 int output_size, float std)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      output_size_(output_size),
      std_(std) {
  // Initialize hyperparameters
  learning_rate = 1e-4;
  lambda = 0.5;
  mu = 0.99;
  neural_statistics.resize(hidden_size_);
  H_.resize(hidden_size_);
  initializeW();
}

float SimpleNeuralNet::loss(const std::vector<std::vector<float>>& images,
                            const std::vector<int>& labels,
                            const std::vector<size_t>& batch_idx) {
  // Reset gradience between batches
  dW1.fill(0.f);
  dW2.fill(0.f);
  std::fill(db1.begin(), db1.end(), 0.f);
  std::fill(db2.begin(), db2.end(), 0.f);

  // ************************
  //  Compute loss for batch
  // ************************
  int N = batch_idx.size();  // Batch size
  float L = 0;               // Accumulated loss for the batch
  for (int i = 0; i < N; ++i) {
    L += loss_one_image(images[batch_idx[i]], labels[batch_idx[i]]);
  }
  L /= N;

  // Regularize loss
  float W1_reg(0), W2_reg(0);
  for (int i = 0; i < input_size_; ++i) {
    for (int h = 0; h < hidden_size_; ++h) {
      W1_reg += W1(h, i) * W1(h, i);
    }
  }
  for (int s = 0; s < output_size_; ++s) {
    for (int h = 0; h < hidden_size_; ++h) {
      W2_reg += W2(s, h) * W2(s, h);
    }
  }

  L += 0.5 * lambda * W1_reg + 0.5 * lambda * W2_reg;

  // Add gradient regularization
  for (int s = 0; s < output_size_; ++s) {
    for (int h = 0; h < hidden_size_; ++h) {
      dW2(s, h) = dW2(s, h) / static_cast<float>(N) + lambda * W2(s, h);
    }
  }
  for (int i = 0; i < input_size_; ++i) {
    for (int h = 0; h < hidden_size_; ++h) {
      dW1(h, i) = dW1(h, i) / static_cast<float>(N) + lambda * W1(h, i);
    }
  }

  for (float& i : db2) {
    i /= static_cast<float>(N);
  }

  for (float& i : db1) {
    i /= static_cast<float>(N);
  }

  // **********************
  //  Update weights
  // **********************
  t++;
  for (int s = 0; s < output_size_; ++s) {
    for (int h = 0; h < hidden_size_; ++h) {
      // Adam
      mW2(s, h) = beta1 * mW2(s, h) + (1 - beta1) * dW2(s, h);
      vW2(s, h) = beta2 * vW2(s, h) + (1 - beta2) * (dW2(s, h) * dW2(s, h));
      // Bias correction
      float mb = mW2(s, h) / (1 - std::pow(beta1, t));
      float vb = vW2(s, h) / (1 - std::pow(beta2, t));
      W2(s, h) += -learning_rate * mb / (std::sqrt(vb) + 1.0e-8);
      // Momentum
      // vW2(s, h) = mu*vW2(s, h) - learning_rate*dW2(s, h);
      // W2(s,h) += vW2(s, h);
      // SGD
      // W2(s,h) += -learning_rate*dW2(s, h);
    }
  }
  for (int i = 0; i < b2.size(); ++i) {
    // Adam
    mb2[i] = beta1 * mb2[i] + (1 - beta1) * db2[i];
    vb2[i] = beta2 * vb2[i] + (1 - beta2) * (db2[i] * db2[i]);
    // Bias correction
    float mb = mb2[i] / (1 - std::pow(beta1, t));
    float vb = vb2[i] / (1 - std::pow(beta2, t));
    b2[i] += -learning_rate * mb / (std::sqrt(vb) + 1.0e-8);
    // Momentum
    // vb2[i] = mu*vb2[i] - learning_rate*db2[i];
    // b2[i] += vb2[i];
    // SGD
    // b2[i] += - learning_rate*db2[i];
  }
  for (int i = 0; i < input_size_; ++i) {
    for (int h = 0; h < hidden_size_; ++h) {
      // Adam
      mW1(h, i) = beta1 * mW1(h, i) + (1 - beta1) * dW1(h, i);
      vW1(h, i) = beta2 * vW1(h, i) + (1 - beta2) * (dW1(h, i) * dW1(h, i));
      // Bias correction
      float mb = mW1(h, i) / (1 - std::pow(beta1, t));
      float vb = vW1(h, i) / (1 - std::pow(beta2, t));
      W1(h, i) += -learning_rate * mb / (std::sqrt(vb) + 1.0e-8);
      // Momentum
      // vW1(h,i) = mu*vW1(h,i) - learning_rate*dW1(h,i);
      // W1(h,i) += vW1(h,i);
      // SGD
      // W1(h,i) += - learning_rate*dW1(h,i);
    }
  }
  for (int i = 0; i < b1.size(); ++i) {
    // Adam
    mb1[i] = beta1 * mb1[i] + (1 - beta1) * db1[i];
    vb1[i] = beta2 * vb1[i] + (1 - beta2) * (db1[i] * db1[i]);
    // Bias correction
    float mb = mb1[i] / (1 - std::pow(beta1, t));
    float vb = vb1[i] / (1 - std::pow(beta2, t));
    b1[i] += -learning_rate * mb / (std::sqrt(vb) + 1.0e-8);
    // Momentum
    // vb1[i] = mu*vb1[i] - learning_rate*db1[i];
    // b1[i] += vb1[i];
    // SGD
    // b1[i] += - learning_rate*db1[i];
  }

  return L;
}

float SimpleNeuralNet::loss_one_image(const std::vector<float>& image,
                                      const int& y) {
  // **********************
  //  Compute forward pass
  // **********************
  // H = W1*x + b1
  std::vector<float> H(hidden_size_);
  for (int h = 0; h < hidden_size_; ++h) {
    for (int i = 0; i < input_size_; ++i) {
      H[h] += W1(h, i) * image[i] + b1[h];
    }
  }

  // ReLu and it's derivative
  std::vector<float> dmax(hidden_size_, 1.f);
  for (int h = 0; h < hidden_size_; ++h) {
    // ReLU
    if (H[h] < 0.0f) {
      H[h] = 0.0f;
      dmax[h] = 0.0f;
    }
    // Uncomment for leaky  ReLU
    // if(H[h] < 0.01f*H[h]) {
    //     H[h] = 0.01f*H[h];
    //     dmax[h] = 0.01f;
    // }
  }

  // S = W2*H + b2
  std::fill(S.begin(), S.end(), 0.f);
  for (int s = 0; s < output_size_; ++s) {
    for (int h = 0; h < hidden_size_; ++h) {
      S[s] += W2(s, h) * H[h] + b2[s];
    }
  }

  // ***********************
  //  Compute loss function
  // ***********************

  // Subtract from scores max to make our computations
  // numerically stable
  std::vector<float> nscores(S);
  float max = *std::max_element(nscores.begin(), nscores.end());
  for (auto& a : nscores) a -= max;

  // Our data is unnormalized log probabilities
  // Exponentiate it
  std::vector<float> power(nscores);
  for (auto& a : power) {
    a = std::exp(a);
  }

  // Normalize it
  float normalizer = std::accumulate(power.begin(), power.end(), 0.f);
  std::vector<float> prob(power);
  for (auto& a : prob) {
    a /= normalizer;
  }

  float loss = -log(prob[y]);

  // ***********************
  //  Compute backward pass
  // ***********************

  // Compute gradient for loss
  prob[y] -= 1;

  // Propagate it into W2
  // dW2 = H*dscores
  for (int s = 0; s < output_size_; ++s) {
    for (int h = 0; h < hidden_size_; ++h) {
      dW2(s, h) += H[h] * prob[s];
    }
  }

  // Propagate it into b2
  // db2 = 1*dscores
  for (int s = 0; s < output_size_; ++s) {
    db2[s] += prob[s];
  }

  // Propagate gradient further through nonlinearity
  // dhidden = W2*dscores if H was non-negative
  std::vector<float> dhidden(hidden_size_, 0.f);
  for (int s = 0; s < output_size_; ++s) {
    for (int h = 0; h < hidden_size_; ++h) {
      dhidden[h] += W2(s, h) * prob[s] * dmax[h];
    }
  }

  // Propagate into W1
  for (int i = 0; i < input_size_; ++i) {
    for (int h = 0; h < hidden_size_; ++h) {
      dW1(h, i) += dhidden[h] * image[i];
    }
  }

  // Propagate into b1
  for (int h = 0; h < hidden_size_; ++h) {
    db1[h] += dhidden[h];
  }

  return loss;
}

int SimpleNeuralNet::inference(const std::vector<float>& image) {
  std::vector<float> scores = inference_scores(image);

  return std::max_element(scores.begin(), scores.end()) - scores.begin();
}

std::vector<float> SimpleNeuralNet::inference_scores(
    const std::vector<float>& image) {
  // **********************
  //  Compute forward pass
  // **********************
  // H = W1*x + b1
  std::vector<float> H(hidden_size_);
  for (int h = 0; h < hidden_size_; ++h) {
    for (int i = 0; i < input_size_; ++i) {
      H[h] += W1(h, i) * image[i] + b1[h];
    }
  }

  // ReLU
  for (int h = 0; h < hidden_size_; ++h) {
    // ReLU
    H[h] = std::max(0.0f, H[h]);
    // Leaky ReLU
    // H[h] = std::max(0.01f*H[h], H[h]);
    neural_statistics[h] += (H[h] > 0);
  }
  this->H_ = H;

  // S = W2*H + b2
  std::fill(S.begin(), S.end(), 0.f);
  for (int s = 0; s < output_size_; ++s) {
    for (int h = 0; h < hidden_size_; ++h) {
      S[s] += W2(s, h) * H[h] + b2[s];
    }
  }

  return S;
}

void SimpleNeuralNet::initializeW() {
  // Create random generator
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<float> distribution(0, std_);

  // Initialize weights and biases
  W1.setSize(hidden_size_, input_size_);
  b1.resize(hidden_size_);
  std::fill(b1.begin(), b1.end(), 0.001);

  // Random init of W1
  for (int x = 0; x < W1.xSize(); ++x) {
    for (int y = 0; y < W1.ySize(); ++y) {
      W1(x, y) = distribution(generator);
    }
  }

  W2.setSize(output_size_, hidden_size_);
  b2.resize(output_size_);
  std::fill(b2.begin(), b2.end(), 0.001);

  // Random init of W2
  for (int x = 0; x < W2.xSize(); ++x) {
    for (int y = 0; y < W2.ySize(); ++y) {
      W2(x, y) = distribution(generator);
    }
  }

  // Initialize gradients
  dW1.setSize(hidden_size_, input_size_);
  dW2.setSize(output_size_, hidden_size_);
  dW1.fill(0.f);
  dW2.fill(0.f);

  db1.resize(hidden_size_);
  db2.resize(output_size_);
  std::fill(db1.begin(), db1.end(), 0.f);
  std::fill(db2.begin(), db2.end(), 0.f);

  // Initialize score
  S.resize(output_size_);

  // Initialize momentum
  vW1.setSize(hidden_size_, input_size_);
  vW2.setSize(output_size_, hidden_size_);
  vW1.fill(0.f);
  vW2.fill(0.f);
  vb1.resize(hidden_size_);
  vb2.resize(output_size_);
  std::fill(vb1.begin(), vb1.end(), 0.f);
  std::fill(vb2.begin(), vb2.end(), 0.f);

  // Initialize parameters for Adam
  mW1.setSize(hidden_size_, input_size_);
  mW2.setSize(output_size_, hidden_size_);
  mW1.fill(0.f);
  mW2.fill(0.f);
  mb1.resize(hidden_size_);
  mb2.resize(output_size_);
  std::fill(mb1.begin(), mb1.end(), 0.f);
  std::fill(mb2.begin(), mb2.end(), 0.f);
  t = 0;
  beta1 = 0.9;
  beta2 = 0.999;
}

bool SimpleNeuralNet::saveWeights(const std::string& filename) {
  std::ofstream file;
  file.open(filename, std::ofstream::binary);

  if (!file.is_open()) {
    std::cout << "Couldn't open the file " << filename << std::endl;
    return false;
  }

  size_t size;

  // Save W1
  size = W1.xSize();
  file.write(reinterpret_cast<char*>(&size), sizeof(size));
  size = W1.ySize();
  file.write(reinterpret_cast<char*>(&size), sizeof(size));
  file.write(reinterpret_cast<char*>(W1.data()), W1.size() * sizeof(float));

  // Save W2
  size = W2.xSize();
  file.write(reinterpret_cast<char*>(&size), sizeof(size));
  size = W2.ySize();
  file.write(reinterpret_cast<char*>(&size), sizeof(size));
  file.write(reinterpret_cast<char*>(W2.data()), W2.size() * sizeof(float));

  // Save b1
  size = b1.size();
  file.write(reinterpret_cast<char*>(&size), sizeof(size));
  file.write(reinterpret_cast<char*>(b1.data()), size * sizeof(float));

  // Save b2
  size = b2.size();
  file.write(reinterpret_cast<char*>(&size), sizeof(size));
  file.write(reinterpret_cast<char*>(b2.data()), size * sizeof(float));

  file.close();
  return true;
}

bool SimpleNeuralNet::loadWeights(const std::string& filename) {
  std::ifstream file;
  file.open(filename, std::ifstream::binary);

  if (!file.is_open()) {
    std::cout << "Couldn't open the file " << filename << std::endl;
    return false;
  }

  size_t xSize, ySize, size;

  // read W1
  file.read(reinterpret_cast<char*>(&xSize), sizeof(xSize));
  file.read(reinterpret_cast<char*>(&ySize), sizeof(ySize));
  W1.setSize(xSize, ySize);
  file.read(reinterpret_cast<char*>(W1.data()), xSize * ySize * sizeof(float));

  // read W2
  file.read(reinterpret_cast<char*>(&xSize), sizeof(xSize));
  file.read(reinterpret_cast<char*>(&ySize), sizeof(ySize));
  W2.setSize(xSize, ySize);
  file.read(reinterpret_cast<char*>(W2.data()), xSize * ySize * sizeof(float));

  // read b1
  file.read(reinterpret_cast<char*>(&size), sizeof(size));
  b1.resize(size);
  file.read(reinterpret_cast<char*>(b1.data()), size * sizeof(float));

  // read b2
  file.read(reinterpret_cast<char*>(&size), sizeof(size));
  b2.resize(size);
  file.read(reinterpret_cast<char*>(b2.data()), size * sizeof(float));

  file.close();
  return true;
}

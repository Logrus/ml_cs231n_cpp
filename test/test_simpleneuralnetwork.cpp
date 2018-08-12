#undef NDEBUG  // Do assert always
#include <assert.h>
#include <classifiers/simpleneuralnet.h>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
  // ==== Parameters for tests ======

  const float epsilon = 1.e-6;

  const std::vector<std::vector<float> > W1 = {
      {0.17640523, 0.04001572, 0.0978738, 0.22408932, 0.1867558, -0.09772779, 0.09500884,
       -0.01513572, -0.01032189, 0.04105985},
      {0.01440436, 0.14542735, 0.07610377, 0.0121675, 0.04438632, 0.03336743, 0.14940791,
       -0.02051583, 0.03130677, -0.08540957},
      {-0.25529898, 0.06536186, 0.08644362, -0.0742165, 0.22697546, -0.14543657, 0.00457585,
       -0.01871839, 0.15327792, 0.14693588},
      {0.01549474, 0.03781625, -0.08877857, -0.19807965, -0.03479121, 0.0156349, 0.12302907,
       0.12023798, -0.03873268, -0.03023028}};

  const std::vector<std::vector<float> > W2 = {
      {-0.1048553, -0.14200179, -0.17062702},  {0.19507754, -0.05096522, -0.04380743},
      {-0.12527954, 0.07774904, -0.16138978},  {-0.02127403, -0.08954666, 0.03869025},
      {-0.05108051, -0.11806322, -0.00281822}, {0.04283319, 0.00665172, 0.03024719},
      {-0.06343221, -0.03627412, -0.06724604}, {-0.03595532, -0.08131463, -0.17262826},
      {0.01774261, -0.04017809, -0.16301983},  {0.04627823, -0.09072984, 0.00519454}};

  const std::vector<std::vector<float> > dW1 = {
      {-9.68499043e-02, 4.00157208e-03, -2.05517828e-01, 1.87986352e-01, 1.60531645e-01,
       -9.77277880e-03, 9.50088418e-03, 2.68884345e-03, -3.01022811e-02, -5.67802801e-03},
      {4.45595008e-02, 1.45427351e-02, 6.95515502e-01, -2.88616327e-01, -2.66986989e-01,
       3.33674327e-03, 1.49407907e-02, 1.93435586e-02, -6.54700997e-02, -5.32928651e-01},
      {1.16977821e-02, 6.53618595e-03, -2.31623550e-01, -6.26390355e-02, -1.41638971e-03,
       -1.45436567e-02, 4.57585173e-04, -2.90067077e-03, 5.35668029e-01, 3.69731998e-01},
      {7.71766403e-02, 3.78162520e-03, -3.13778323e-01, 2.26868568e-01, 2.06678709e-01,
       1.56348969e-03, 1.23029068e-02, -2.18055786e-03, -6.78943040e-01, 9.85573015e-02}};

  const std::vector<std::vector<float> > dW2 = {{-5.13764691e-01, 1.67232930e-01, 3.04783350e-01},
                                                {1.95077540e-02, -5.09652182e-03, -4.38074302e-03},
                                                {2.92229174e-01, 1.18896894e-01, -4.32018096e-01},
                                                {-2.33121075e-01, 1.86288200e-01, 3.96198312e-02},
                                                {7.78174796e-01, -3.54233027e-01, -4.41137965e-01},
                                                {4.28331871e-03, 6.65172224e-04, 3.02471898e-03},
                                                {-6.34322094e-03, -3.62741166e-03, -6.72460448e-03},
                                                {6.17281609e-02, -1.29900489e-01, 3.91825079e-02},
                                                {6.32053946e-01, -8.44023525e-02, -5.66197124e-01},
                                                {8.89334995e-01, -6.04709349e-01, -2.88551353e-01}};

  const std::vector<float> db1 = {-0.0070484, 0., 0.00310494, -0.0072399, -0.00573377,
                                  0.,         0., -0.0024372, 0.04121605, 0.02236176};
  const std::vector<float> db2 = {0.2099691, -0.1431905, -0.0667786};

  const std::vector<std::vector<float> > inputs = {
      {16.24345364, -6.11756414, -5.28171752, -10.72968622},
      {8.65407629, -23.01538697, 17.44811764, -7.61206901},
      {3.19039096, -2.49370375, 14.62107937, -20.60140709},
      {-3.22417204, -3.84054355, 11.33769442, -10.99891267},
      {-1.72428208, -8.77858418, 0.42213747, 5.82815214}};

  const std::vector<int> labels = {0, 1, 2, 2, 1};

  const std::vector<std::vector<float> > expected_scores = {
      {-0.81233741, -1.27654624, -0.70335995},
      {-0.17129677, -1.18803311, -0.47310444},
      {-0.51590475, -1.01354314, -0.8504215},
      {-0.15419291, -0.48629638, -0.52901952},
      {-0.00618733, -0.12435261, -0.15226949}};

  const std::vector<size_t> batch_idx = {0, 1, 2, 3, 4};

  const double correct_loss = 1.30378789133;

  // ============ Test ===============
  SimpleNeuralNet net(4, 10, 3, 0.001);

  assert(net.W1.ySize() == W1.size());
  assert(net.W1.xSize() == W1[0].size());
  assert(net.W2.ySize() == W2.size());
  assert(net.W2.xSize() == W2[0].size());

  // Set biases
  std::fill(net.b1.begin(), net.b1.end(), 0.f);
  std::fill(net.b2.begin(), net.b2.end(), 0.f);

  // Set weights
  std::cout << "W1: " << std::endl;
  for (int y = 0; y < net.W1.ySize(); ++y) {
    for (int x = 0; x < net.W1.xSize(); ++x) {
      net.W1(x, y) = W1[y][x];
      std::cout << std::setw(10) << net.W1(x, y) << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "W2: " << std::endl;
  for (int y = 0; y < net.W2.ySize(); ++y) {
    for (int x = 0; x < net.W2.xSize(); ++x) {
      net.W2(x, y) = W2[y][x];
      std::cout << std::setw(10) << net.W2(x, y) << " ";
    }
    std::cout << std::endl;
  }

  // Test scores in inference
  std::cout << "Testing scores in inference..." << std::flush;
  for (int i = 0; i < inputs.size(); ++i) {
    std::vector<float> scores = net.inference_scores(inputs[i]);
    for (int j = 0; j < scores.size(); ++j) {
      assert((scores[j] - expected_scores[i][j]) < epsilon);
    }
  }
  std::cout << "passed." << std::endl;

  // Test scores in loss
  std::cout << "Testing scores in loss..." << std::flush;
  for (int i = 0; i < inputs.size(); ++i) {
    net.loss_one_image(inputs[i], labels[i]);
    for (int j = 0; j < net.S.size(); ++j) {
      assert((net.S[j] - expected_scores[i][j]) < epsilon);
    }
  }
  std::cout << "passed." << std::endl;

  // Test loss
  std::cout << "Testing loss..." << std::flush;
  net.lambda = 0.1f;
  float loss = net.loss(inputs, labels, batch_idx);
  assert((correct_loss - loss) < epsilon);
  std::cout << "passed." << std::endl;

  // Check gradients
  std::cout << "Testing db2..." << std::flush;
  for (int i = 0; i < db2.size(); ++i) {
    assert((net.db2[i] - db2[i]) < epsilon);
  }
  std::cout << "passed." << std::endl;

  std::cout << "Testing db1..." << std::flush;
  for (int i = 0; i < db1.size(); ++i) {
    assert((net.db1[i] - db1[i]) < epsilon);
  }
  std::cout << "passed." << std::endl;

  std::cout << "Testing dW1..." << std::flush;
  for (int y = 0; y < net.dW1.ySize(); ++y) {
    for (int x = 0; x < net.dW1.xSize(); ++x) {
      assert((net.dW1(x, y) - dW1[y][x]) < epsilon);
    }
  }
  std::cout << "passed." << std::endl;

  std::cout << "Testing dW2..." << std::flush;
  for (int y = 0; y < net.dW2.ySize(); ++y) {
    for (int x = 0; x < net.dW2.xSize(); ++x) {
      assert((net.dW2(x, y) - dW2[y][x]) < epsilon);
    }
  }
  std::cout << "passed." << std::endl;

  std::cout << "All tests passed! " << std::endl;
  return 0;
}

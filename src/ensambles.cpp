#include <classifiers/cifar_reader.h>
#include <classifiers/simpleneuralnet.h>
#include <iostream>

int main() {
  // Load dataset
  CIFAR10Reader trainset;

  std::vector<std::string> names = {"../CIFAR10/data_batch_1.bin", "../CIFAR10/data_batch_2.bin",
                                    "../CIFAR10/data_batch_3.bin", "../CIFAR10/data_batch_4.bin",
                                    "../CIFAR10/data_batch_5.bin"};
  for (int i = 0; i < names.size(); ++i) {
    trainset.read_bin(names[i], false);
  }

  // Demean test set
  trainset.compute_mean();

  CIFAR10Reader testset;
  testset.read_bin("../CIFAR10/test_batch.bin", false);

  // Demean training set
  testset.setMeanImage(trainset.mean_image());
  testset.demean();

  SimpleNeuralNet net1(3072, 70, 10, 0.0001);
  SimpleNeuralNet net2(3072, 70, 10, 0.0001);
  SimpleNeuralNet net3(3072, 70, 10, 0.0001);
  SimpleNeuralNet net4(3072, 70, 10, 0.0001);

  net1.loadWeights("weights1.dat");
  net2.loadWeights("weights2.dat");
  net3.loadWeights("weights3.dat");
  net4.loadWeights("weights4.dat");

  // Evaluate accuracy for net1
  int correct = 0;
  int total = 0;
  for (int i = 0; i < testset.images().size(); ++i) {
    int label = net1.inference(testset.images()[i]);
    if (label == testset.labels()[i]) correct++;
    total++;
  }
  std::cout << "Net1 acc: " << correct / static_cast<float>(total) << std::endl;

  // Evaluate accuracy for net2
  correct = 0;
  total = 0;
  for (int i = 0; i < testset.images().size(); ++i) {
    int label = net2.inference(testset.images()[i]);
    if (label == testset.labels()[i]) correct++;
    total++;
  }
  std::cout << "Net2 acc: " << correct / static_cast<float>(total) << std::endl;

  // Evaluate accuracy for net3
  correct = 0;
  total = 0;
  for (int i = 0; i < testset.images().size(); ++i) {
    int label = net3.inference(testset.images()[i]);
    if (label == testset.labels()[i]) correct++;
    total++;
  }
  std::cout << "Net3 acc: " << correct / static_cast<float>(total) << std::endl;

  // Evaluate accuracy for net4
  correct = 0;
  total = 0;
  for (int i = 0; i < testset.images().size(); ++i) {
    int label = net4.inference(testset.images()[i]);
    if (label == testset.labels()[i]) correct++;
    total++;
  }
  std::cout << "Net4 acc: " << correct / static_cast<float>(total) << std::endl;

  // Evaluate accuracy for ensamble
  SimpleNeuralNet ens(3072, 70, 10, 0.0001);
  ens.loadWeights("weights2.dat");
  // average parameter b1
  for (int i = 0; i < net1.b1.size(); ++i) {
    ens.b1[i] = (net1.b1[i] + net2.b1[i] + net3.b1[i] + net4.b1[i]) / 4.0;
  }
  // average parameter b2
  for (int i = 0; i < net1.b2.size(); ++i) {
    ens.b2[i] = (net1.b2[i] + net2.b2[i] + net3.b2[i] + net4.b2[i]) / 4.0;
  }
  // average parameter W1
  for (int x = 0; x < net1.W1.xSize(); ++x) {
    for (int y = 0; y < net1.W1.ySize(); ++y) {
      ens.W1(x, y) = (net1.W1(x, y) + net2.W1(x, y) + net3.W1(x, y) + net4.W1(x, y)) / 4.0;
    }
  }
  // average parameter W2
  for (int x = 0; x < net1.W2.xSize(); ++x) {
    for (int y = 0; y < net1.W2.ySize(); ++y) {
      net3.W2(x, y) = (net1.W2(x, y) + net2.W2(x, y) + net3.W2(x, y) + net4.W2(x, y)) / 4.0;
    }
  }

  // Evaluate accuracy for net3
  correct = 0;
  total = 0;
  for (int i = 0; i < testset.images().size(); ++i) {
    int label = ens.inference(testset.images()[i]);
    if (label == testset.labels()[i]) correct++;
    total++;
  }
  std::cout << "Weight ensamble acc: " << correct / static_cast<float>(total) << std::endl;

  correct = 0;
  total = 0;
  for (int i = 0; i < testset.images().size(); ++i) {
    auto scores1 = net1.inference_scores(testset.images()[i]);
    auto scores2 = net2.inference_scores(testset.images()[i]);
    auto scores3 = net3.inference_scores(testset.images()[i]);
    auto scores4 = net4.inference_scores(testset.images()[i]);

    auto avg(scores1);
    for (int j = 0; j < avg.size(); ++j) {
      avg[j] = (scores1[j] + scores2[j] + scores3[j] + scores4[j]) / 4.0;
    }
    int label = std::max_element(avg.begin(), avg.end()) - avg.begin();

    if (label == testset.labels()[i]) correct++;
    total++;
  }
  std::cout << "Ensamble acc: " << correct / static_cast<float>(total) << std::endl;

  return 0;
}

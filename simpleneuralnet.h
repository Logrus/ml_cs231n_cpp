#ifndef SIMPLENEURALNET_H
#define SIMPLENEURALNET_H
#include <vector>

typedef std::vector< std::vector<float> > vvfloat;
typedef std::vector<int> vint;

class SimpleNeuralNet
{
public:
    SimpleNeuralNet(int input_size, int hidden_size, int output_size, float std);

    float loss(const vvfloat &images, const vint &labels);

};

#endif // SIMPLENEURALNET_H

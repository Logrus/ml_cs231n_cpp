#include "linearsvm.h"
LinearSVM::LinearSVM(int classes, int dimentionality) :
    C(classes),
    D(dimentionality),
    W(classes,dimentionality),
    dW(classes,dimentionality),
    scores(classes,1),
    lambda(0.5),
    learning_rate(0.000001)
{

    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0,0.00001);

    // Randomly initialize weights
    for(int i=0; i < W.size(); ++i){
        W.data[i] = distribution(generator);
    }
}

float LinearSVM::L2W_reg(){
    float sum = 0;
    for (int i = 0; i < W.size(); ++i){
        sum += W.data[i] * W.data[i];
    }
    return sum;
}

void LinearSVM::updateWeights(){
    for (int i=0; i<W.size(); ++i){
        W.data[i] += -learning_rate*dW.data[i];
        //std::cout << "Weight " << i << " = " << W.data[i] << std::endl;
    }
}

float LinearSVM::loss_one_image(const std::vector<int> &image, const int &y){

    // Reset scores
    std::fill(scores.data.begin(), scores.data.end(), 0.0);

    // scores = W*x
    for(int c=0; c < C; ++c){
        for(int d=0; d < D; ++d){
            scores.data[c] += image[d]*W(c,d);
        }
    }
    count = 0;
    float loss = 0;
    for (int c=0; c < C; ++c)
    {
        if( c == y ) continue;
        float margin = scores(c,1) - scores(y,1) + 1;
        if (margin > 0.0f){
            loss += margin;
            count++;
        }
    }

    // Update gradient
    for(int c=0; c<C; ++c){
        for (int d=0; d<D; ++d){
            if (c == y){
                dW(c,d) += -image[d]*count + lambda*W(c,d);
            } else {
                dW(c,d) += (scores.data[c]>0) * image[d]  + lambda*W(c,d);
            }
        }
    }

    return loss;
}

float LinearSVM::loss(const std::vector< std::vector<int> > &images, const std::vector<int> &labels, int from, int to)
{
    // Reset gradient
    std::fill(dW.data.begin(), dW.data.end(), 0.0);

    // Compute scores for all images
    float L = 0;
    int N = to-from; // N images in dataset
    for(int i=from; i<to; ++i){
        L += loss_one_image(images[i], labels[i]);
    }
    L /= N;
    L += lambda * L2W_reg();
    std::cout << L << std::endl;
    updateWeights();
    return L;
}

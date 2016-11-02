#include "linearsvm.h"
LinearSVM::LinearSVM(int classes, int dimentionality) :
    C(classes),
    D(dimentionality),
    W(classes,dimentionality),
    dW(classes,dimentionality),
    lambda(0.5),
    learning_rate(1.0e-10)
{

    // Randomly initialize weights
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0,0.00001);
    for(int i=0; i < W.size(); ++i){
        W(i) = distribution(generator);
    }

    // Initialize gradient
    std::fill(dW.data.begin(), dW.data.end(), 0.0);
}

float LinearSVM::L2W_reg(){
    float sum = 0;
    for (int i = 0; i < W.size(); ++i){
        sum += W(i) * W(i);
    }
    return sum;
}

float LinearSVM::loss_one_image(const std::vector<int> &image, const int &y){

    assert(image.size() == 3073);

    Matrix scores(C, 1);

    // Compute scores
    // scores = W*x
    for(int c=0; c<C; ++c){
        for(int d=0; d<D; ++d){
            scores(c) += image[d]*W(c,d);
        }
    }

    // Compute loss
    int count = 0;
    float loss = 0;
    for (int j=0; j<C; ++j)
    {
        if( j == y ) continue;
        float margin = scores(j) - scores(y) + 1;
        if (margin > 0.0f){
            loss += margin;
            count++;
        }
    }

    // Compute gradient update
    for(int c=0; c<C; ++c){
        for (int d=0; d<D; ++d){
            if (c == y){
                dW(c,d) += -count*image[d];
            } else {
                dW(c,d) += (scores(c)>0)*image[d];
            }
        }
    }

    return loss;
}

float LinearSVM::loss(const std::vector< std::vector<int> > &images, const std::vector<int> &labels, int from, int to)
{
    assert(images.size() == 60000);

    // Reset gradient
    std::fill(dW.data.begin(), dW.data.end(), 0.0);

    // Compute loss for all images
    float L = 0;
    int N = to-from; // N images in dataset
    for(int i=from; i<to; ++i){
        L += loss_one_image(images[i], labels[i]);
    }
    L /= N;
    L += 0.5 * lambda * L2W_reg();
    std::cout << L << std::endl;

    // Normalize and regularize gradient
    for (int i=0; i<dW.size(); ++i){
        dW(i) += dW(i)/static_cast<float>(N) + lambda*W(i);
    }

    // Update weights
    for (int i=0; i<W.size(); ++i){
        W(i) += -learning_rate*dW(i);
    }

    return L;
}

int LinearSVM::inference(const std::vector<int> &image){

    Matrix scores(C,1);

    // scores = W*x
    for(int c=0; c < C; ++c){
        for(int d=0; d < D; ++d){
            scores(c) += image[d]*W(c,d);
        }
    }

    return std::max_element(scores.data.begin(), scores.data.end()) - scores.data.begin();
}

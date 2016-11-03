#include "classifier.h"

Classifier::Classifier(int classes, int dimentionality) :
    C(classes),
    D(dimentionality),
    lambda(0.5),
    learning_rate(1.0e-8)
{
    W.setSize(classes, dimentionality);
    dW.setSize(classes, dimentionality);

    // Initialize gradient
    dW.fill(0.0);
}

void Classifier::copyW(const CMatrix<float> inW){
    W = inW;
}

std::vector<float> Classifier::scores(const std::vector<float> &image){

    std::vector<float> scores(10, 0);

    // scores = W*x
    for(int c=0; c < C; ++c){
        for(int d=0; d < D; ++d){
            scores[c] += W(c,d)*image[d];
        }
    }

    return scores;
}

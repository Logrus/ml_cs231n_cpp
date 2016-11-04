#include "linearsvm.h"
LinearSVM::LinearSVM(int classes, int dimentionality) :
    Classifier(classes, dimentionality)
{
}

float LinearSVM::L2W_reg(){
    float sum = 0;
    for (int x = 0; x < W.xSize(); ++x)
        for (int y = 0; y < W.ySize(); ++y){
            sum += W(x,y) * W(x,y);
        }
    return sum;
}

float LinearSVM::loss_one_image(const std::vector<float> &image, const int &y){

    assert(image.size() == 3073);

    std::vector<float> scores(10, 0);

    // Compute scores
    // scores = W*x
    for(int c=0; c<C; ++c){
        for(int d=0; d<D; ++d){
            scores[c] += W(c,d)*image[d];
        }
    }

    // Compute loss
    float loss = 0;
    int counter = 0;
    std::vector<float> margins(10,0);
    for (int j=0; j<C; ++j)
    {
        if(j==y) continue;
        float margin = scores[j] - scores[y] + 1;

        loss += std::max(0.f,margin);

        // Compute gradient
        if( margin > 0 )
        {
            for (int d=0; d<D; ++d){
                dW(y,d) -= image[d];
                dW(j,d) += image[d];
            }
        }
    }

    return loss;
}

float LinearSVM::loss(const std::vector< std::vector<float> > &images, const std::vector<int> &labels, int from, int to)
{
    assert(images.size() == 50000);
    assert(C == 10);
    assert(D == 3073);

    //std::cout << "From " << from << " to " << to << " size " << to-from << std::endl;

    // Reset gradient
    dW.fill(0.0);

    // Compute loss for all images
    float L = 0;
    int N = to-from; // N images in batch
    for(int i=from; i<to; ++i){
        L += loss_one_image(images[i], labels[i]);
    }
    L /= N;
    L += 0.5 * lambda * L2W_reg();

    // Normalize and regularize gradient
    for (int x=0; x< dW.xSize(); ++x){
        for (int y=0; y < dW.ySize(); ++y){
            dW(x,y) = dW(x,y)/static_cast<float>(N) + lambda*W(x,y);
        }
    }

    // Update weights
    for (int x=0; x<W.xSize(); ++x){
        for (int y=0; y<W.ySize(); ++y){
            W(x, y) -= learning_rate*dW(x, y);
        }
    }

    return L;
}

int LinearSVM::inference(const std::vector<float> &image){

    std::vector<float> scores(10, 0);

    // scores = W*x
    for(int c=0; c < C; ++c){
        for(int d=0; d < D; ++d){
            scores[c] += W(c,d)*image[d];
        }
    }

    return std::max_element(scores.begin(), scores.end()) - scores.begin();
}

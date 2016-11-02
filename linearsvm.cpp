#include "linearsvm.h"
LinearSVM::LinearSVM(int classes, int dimentionality) :
    C(classes),
    D(dimentionality),
    lambda(0.5),
    learning_rate(0.00000000001)
{
    W.setSize(classes, dimentionality);
    dW.setSize(classes, dimentionality);

    // Randomly initialize weights
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0,0.00001);
    for(int x=0; x < W.xSize(); ++x){
        for(int y=0; y < W.ySize(); ++y)
        {
            W(x,y) = distribution(generator);
            if(y==W.ySize()-1){ //make the weight for bias positive (better for initialization)
                W(x,y)=fabs(W(x,y));
            }
        }
    }

    // Initialize gradient
    dW.fill(0.0);
}

float LinearSVM::L2W_reg(){
    float sum = 0;
    for (int x = 0; x < W.xSize(); ++x)
        for (int y = 0; y < W.ySize(); ++y){
            sum += W(x,y) * W(x,y);
        }
    return sum;
}

float LinearSVM::loss_one_image(const std::vector<int> &image, const int &y){

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
        margins[j] = std::max(0.f, scores[j] - scores[y] + 1);
        counter += (margins[j]>0);
        loss += margins[j];

    }

    // Compute gradient
    for (int j=0; j<C; ++j)
        for (int d=0; d<D; ++d){
            if(j==y){
                dW(j,d) += -image[d]*counter;
            } else if(j!=y) {
                dW(j,d) += (margins[j]>0)*image[d];
            }
        }


    return loss;
}

float LinearSVM::loss(const std::vector< std::vector<int> > &images, const std::vector<int> &labels, int from, int to)
{
    assert(images.size() == 60000);

    // Reset gradient
    dW.fill(0.0);

    // Compute loss for all images
    float L = 0;
    int N = 100; // N images in batch
    for(int i=0; i<N; ++i){
        L += loss_one_image(images[i], labels[i]);
    }
    L /= N;
    L += 0.5 * lambda * L2W_reg();
    //std::cout << "Loss: " << L << std::endl;

    // Normalize and regularize gradient
    for (int x=0; x< dW.xSize(); ++x){
        for (int y=0; y < dW.ySize(); ++y){
            dW(x,y) = dW(x,y)/static_cast<float>(N); // + lambda*W(x,y);
        }
    }

    // Update weights
    for (int x=0; x<W.xSize(); ++x){
        for (int y=0; y<W.ySize(); ++y){
            W(x, y) -= learning_rate*dW(x, y);
            //std::cout << "W (" << x << ", " << y << "): " <<  W(x, y) << " ";
        }
        //std::cout << std::endl;
    }

    return L;
}

int LinearSVM::inference(const std::vector<int> &image){

    std::vector<float> scores(10, 0);

    // scores = W*x
    for(int c=0; c < C; ++c){
        for(int d=0; d < D; ++d){
            scores[c] += W(c,d)*image[d];
        }
    }

    return std::max_element(scores.begin(), scores.end()) - scores.begin();
}

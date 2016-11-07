#include "cifar_reader.h"

bool CIFAR10Reader::read_bin(std::string filepath){
  
  std::cout << "Reading " << filepath << std::endl;

  std::ifstream file(filepath.c_str(), std::ios::binary);

  if (!file.is_open()){
      return false;
  }

  int number_of_images = 10000;
  int n_rows = 32;
  int n_cols = 32;
  for(int i = 0; i < number_of_images; ++i)
  {
      //read label for the image
      unsigned char tplabel = 0;
      file.read((char*) &tplabel, sizeof(tplabel));
      //push to the vector of labels
      labels_.push_back((int)tplabel);

      std::vector<float> picture;
      for(int channel = 0; channel < 3; ++channel){
          for(int x = 0; x < n_rows; ++x){
              for(int y = 0; y < n_cols; ++y){
                 unsigned  char temp = 0;
                 file.read((char*) &temp, sizeof(temp));
                 picture.push_back((int)temp);
              }
          }
      }
      picture.push_back(1); // Bias trick
      images_.push_back(picture);
  }
  file.close();
  
  images_copy_ = images_;

  return true;
}

void CIFAR10Reader::compute_mean()
{
    std::cout << "Computing mean image" << std::endl;
    // Compute mean image
    mean_image.resize(images_copy_[0].size());
    std::fill(mean_image.begin(), mean_image.end(), 0.0f);
    for(int i=0; i < images_copy_.size(); ++i){
        const std::vector<float> *image = &images_copy_[i];
        for(int d=0; d < image->size(); ++d){
            mean_image[d] += image->at(d);
        }
    }
    for(int d=0; d < mean_image.size(); ++d){
        mean_image[d] /= images_copy_.size();
    }

}

void CIFAR10Reader::compute_std()
{
    std::cout << "Computing std image" << std::endl;
    // Compute mean image
    std_image.resize(images_copy_[0].size());
    std::fill(std_image.begin(), std_image.end(), 0.0f);
    for(int i=0; i < images_copy_.size(); ++i){
        const std::vector<float> *image = &images_copy_[i];
        for(int d=0; d < image->size(); ++d){
            std_image[d] += (image->at(d) - mean_image[d])*(image->at(d) - mean_image[d]);
        }
    }
    for(int d=0; d < mean_image.size(); ++d){
        std_image[d] = sqrt( std_image[d]/images_copy_.size() );
    }
}

void CIFAR10Reader::normalize()
{
    auto minmax = this->minmax();
    float range = minmax.second - minmax.first;
    for(int i=0; i < images_.size(); ++i){
        std::vector<float> *image = &images_[i];
        for(int d=0; d < image->size(); ++d){
            image->at(d) = (image->at(d))/range;
        }
    }
}

void CIFAR10Reader::standardize()
{
    compute_mean();
    compute_std();
    for(int i=0; i < images_copy_.size(); ++i){
        std::vector<float> *image = &images_[i];
        for(int d=0; d < image->size(); ++d){
            image->at(d) = (image->at(d) - mean_image[d])/std_image[d];
        }
    }
}

void CIFAR10Reader::demean()
{
    compute_mean();

    // Demean every image
    for(int i=0; i< images_.size(); ++i){
        std::vector<float> *image = &images_[i];
        for(int d=0; d < image->size(); ++d){
            image->at(d) -= mean_image[d];
        }
    }
    std::cout << "Subtracted mean from every image." << std::endl;
    auto minmax = std::minmax_element(mean_image.begin(), mean_image.end());
    std::cout << "Mean image min " << *minmax.first << " max " << *minmax.second << std::endl;
}

void CIFAR10Reader::reset()
{
    images_ = images_copy_;
}

std::pair<float, float> CIFAR10Reader::minmax()
{
    float gmin = std::numeric_limits<float>::max();
    float gmax = std::numeric_limits<float>::min();
    for(int i=0; i < images_.size(); ++i){
        const std::vector<float> *image = &images_[i];
        auto minmax = std::minmax_element(image->begin(), image->end());
        if (*minmax.first < gmin) gmin = *minmax.first;
        if (*minmax.second > gmax) gmax = *minmax.second;
    }

    return {gmin, gmax};
}

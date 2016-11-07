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

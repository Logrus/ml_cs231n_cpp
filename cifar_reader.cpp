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
      char tplabel;
      file.read(&tplabel, sizeof(tplabel));
      //push to the vector of labels
      labels_.push_back(static_cast<int>(tplabel));

      std::vector<int> picture;
      for(int channel = 0; channel < 3; ++channel){
          for(int x = 0; x < n_rows; ++x){
              for(int y = 0; y < n_cols; ++y){
                 char temp = 0;
                 file.read(&temp, sizeof(temp));
                 picture.push_back(static_cast<int>(temp));
              }
          }
      }
      picture.push_back(1); // Bias trick
      images_.push_back(picture);
  }
  file.close();
  
  return true;
}

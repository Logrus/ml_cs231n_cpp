#include <classifiers/cifar_reader.h>

bool CIFAR10Reader::read_bin(const std::string filepath, bool bias_trick = false) {
  std::cout << "Reading " << filepath << std::endl;

  std::ifstream file(filepath.c_str(), std::ios::binary);

  if (!file.is_open()) {
    return false;
  }

  const int number_of_images = 10000;
  const int n_rows = 32;
  const int n_cols = 32;
  for (int i = 0; i < number_of_images; ++i) {
    // read label for the image
    char tplabel = 0;
    file.read(static_cast<char*>(&tplabel), sizeof(tplabel));
    // push to the vector of labels
    labels_.push_back(static_cast<int>(tplabel));

    std::vector<float> picture;
    for (size_t channel = 0; channel < 3; ++channel) {
      for (size_t x = 0; x < n_rows; ++x) {
        for (size_t y = 0; y < n_cols; ++y) {
          char temp = 0;
          file.read(static_cast<char*>(&temp), sizeof(temp));
          picture.push_back(static_cast<int>(temp));
        }
      }
    }
    if (bias_trick) {
      picture.push_back(1);  // Bias trick
    }
    images_.push_back(picture);
  }
  file.close();

  shuffler.set_nelem(images_.size());

  images_copy_ = images_;

  return true;
}

void CIFAR10Reader::compute_mean() {
  std::cout << "Computing mean image" << std::endl;
  // Compute mean image
  mean_image.resize(images_copy_[0].size());
  std::fill(mean_image.begin(), mean_image.end(), 0.0f);
  for (size_t i = 0; i < images_.size(); ++i) {
    const std::vector<float>& image = images_[i];
    for (size_t d = 0; d < image.size(); ++d) {
      mean_image[d] += image.at(d);
    }
  }
  for (size_t d = 0; d < mean_image.size(); ++d) {
    mean_image[d] /= static_cast<float>(images_copy_.size());
  }
}

void CIFAR10Reader::compute_std() {
  std::cout << "Computing std image" << std::endl;
  // Compute mean image
  std_image.resize(images_copy_[0].size());
  std::fill(std_image.begin(), std_image.end(), 0.0f);
  for (size_t i = 0; i < images_.size(); ++i) {
    const std::vector<float>* image = &images_[i];
    for (size_t d = 0; d < image->size(); ++d) {
      float diff = (image->at(d) - mean_image[d]);
      std_image[d] += diff * diff;
    }
  }
  for (size_t d = 0; d < std_image.size(); ++d) {
    std_image[d] = sqrtf(std_image[d] / images_.size());
  }
}

void CIFAR10Reader::normalize() {
  auto minmax = this->minmax();
  float range = minmax.second - minmax.first;
  for (size_t i = 0; i < images_.size(); ++i) {
    std::vector<float>* image = &images_[i];
    for (size_t d = 0; d < image->size(); ++d) {
      image->at(d) = (image->at(d)) / range;
    }
  }
  state_ = PreprocessingType::NORMALIZED;
}

void CIFAR10Reader::standardize() {
  // compute_mean();
  // compute_std();
  float max = -100;
  float min = 100;
  for (size_t i = 0; i < images_.size(); ++i) {
    //        std::vector<float> *image = &images_[i];
    for (size_t d = 0; d < images_[0].size(); ++d) {
      images_[i][d] -= mean_image[d];
      images_[i][d] /= std_image[d] + 0.0001f;
      if (max < images_[i][d]) max = images_[i][d];
      if (min > images_[i][d]) min = images_[i][d];
    }
  }
  std::cout << "Max! " << max << " min " << min << std::endl;
  state_ = PreprocessingType::STANDARDIZED;
}

void CIFAR10Reader::demean() {
  // compute_mean();

  // Demean every image
  for (size_t i = 0; i < images_.size(); ++i) {
    std::vector<float>* image = &images_[i];
    for (size_t d = 0; d < image->size(); ++d) {
      image->at(d) -= mean_image[d];
    }
  }
  std::cout << "Subtracted mean from every image." << std::endl;
  auto minmax = std::minmax_element(mean_image.begin(), mean_image.end());
  std::cout << "Mean image min " << *minmax.first << " max " << *minmax.second << std::endl;
  state_ = PreprocessingType::DEMEANED;
}

void CIFAR10Reader::reset() { images_ = images_copy_; }

std::pair<float, float> CIFAR10Reader::minmax() {
  float gmin = std::numeric_limits<float>::max();
  float gmax = std::numeric_limits<float>::min();
  for (size_t i = 0; i < images_.size(); ++i) {
    const std::vector<float>* image = &images_[i];
    auto minmax = std::minmax_element(image->begin(), image->end());
    if (*minmax.first < gmin) gmin = *minmax.first;
    if (*minmax.second > gmax) gmax = *minmax.second;
  }

  return {gmin, gmax};
}

std::vector<size_t> CIFAR10Reader::get_batch_idxs(int batch_size) const {
  return shuffler.get_random_indexies(batch_size);
}

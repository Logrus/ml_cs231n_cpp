#include <classifiers/cifar_reader.h>
#include <limits>

namespace {
/// Defined for CIFAR10
constexpr size_t kNumOfChannels = 3;
constexpr int kNumberOfImages = 10000;
constexpr int KNRows = 32;
constexpr int KNCols = 32;
}  // namespace

bool CIFAR10Reader::readBin(const std::string filepath, const bool bias_trick = false) {
  std::cout << "Reading " << filepath << std::endl;

  std::ifstream file(filepath.c_str(), std::ios::binary);

  if (!file.is_open()) {
    return false;
  }

  for (int i = 0; i < kNumberOfImages; ++i) {
    // read label for the image
    unsigned char tplabel = 0;
    file.read(reinterpret_cast<char*>(&tplabel), sizeof(tplabel));
    // push to the vector of labels
    labels_.push_back(static_cast<int>(tplabel));

    std::vector<float> picture;
    for (size_t channel = 0; channel < kNumOfChannels; ++channel) {
      for (size_t x = 0; x < KNRows; ++x) {
        for (size_t y = 0; y < KNCols; ++y) {
          unsigned char temp = 0;
          file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
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

  shuffler.setNelem(images_.size());

  images_copy_ = images_;

  return true;
}

void CIFAR10Reader::computeMean() {
  std::cout << "Computing mean image" << std::endl;
  // Compute mean image
  mean_image_.resize(images_.front().size(), 0.f);

  const float num_images = static_cast<float>(images_.size());
  for (const auto& image : images_) {
    for (size_t pix = 0; pix < image.size(); ++pix) {
      mean_image_[pix] += image.at(pix) / num_images;
    }
  }
}

void CIFAR10Reader::computeStd() {
  std::cout << "Computing std image" << std::endl;
  // Check mean image
  if (mean_image_.empty() || mean_image_.size() != images_.front().size()) {
    std::cerr << "Unable to compute standard deviation since mean image was not computed yet!\n"
                 "Please call compute_mean() before"
              << std::endl;
  }
  std_image_.resize(images_copy_[0].size(), 0.f);
  for (const auto& image : images_) {
    for (size_t pix = 0; pix < image.size(); ++pix) {
      const float diff = (image.at(pix) - mean_image_.at(pix));
      std_image_[pix] += diff * diff;
    }
  }
  for (size_t d = 0; d < std_image_.size(); ++d) {
    std_image_[d] = sqrtf(std_image_[d] / images_.size());
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
  float max = std::numeric_limits<float>::min();
  float min = std::numeric_limits<float>::max();
  for (size_t i = 0; i < images_.size(); ++i) {
    //        std::vector<float> *image = &images_[i];
    for (size_t d = 0; d < images_.front().size(); ++d) {
      images_[i][d] -= mean_image_[d];
      images_[i][d] /= std_image_[d] + 0.0001f;
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
  for (auto& image : images_) {
    for (size_t d = 0; d < image.size(); ++d) {
      image.at(d) -= mean_image_[d];
    }
  }
  std::cout << "Subtracted mean from every image." << std::endl;
  auto minmax = std::minmax_element(mean_image_.begin(), mean_image_.end());
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

std::vector<size_t> CIFAR10Reader::getBatchIdxs(int batch_size) const {
  return shuffler.getRandomIndexies(batch_size);
}

std::vector<float> CIFAR10Reader::undoPreprocessing(
    const std::vector<float>& preprocessed_image) const {
  std::vector<float> result(preprocessed_image.size());
  if (state_ == PreprocessingType::DEMEANED) {
    // We should add mean image back
    for (size_t pix = 0; pix < preprocessed_image.size(); ++pix) {
      result[pix] += preprocessed_image[pix] + mean_image_[pix];
    }
  } else if (state_ == PreprocessingType::NORMALIZED) {
    throw std::string("Undo preprocessing for NORMALIZED images isn't implemented yet");
  } else if (state_ == PreprocessingType::STANDARDIZED) {
    throw std::string("Undo preprocessing for STANDARDIZED images isn't implemented yet");
  } else if (state_ == PreprocessingType::NO_PREPROCESSING) {
    result = preprocessed_image;  // just copy
  }

  return result;
}

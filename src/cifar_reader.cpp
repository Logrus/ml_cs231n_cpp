#include <classifiers/cifar_reader.h>
#include <limits>

namespace {
/// Defined for CIFAR10
constexpr size_t kNumOfChannels = 3;
constexpr int kNumberOfImages = 10000;
constexpr int KNRows = 32;
constexpr int KNCols = 32;
}  // namespace

bool CIFAR10Reader::readBin(const std::string& filepath,
                            const bool bias_trick = false) {
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
    labels_.emplace_back(tplabel);

    std::array<unsigned char, kNumOfChannels * KNRows * KNCols> buffer;
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    if (!file.good()) {
      std::cerr << "Error while reading CIFAR10 dataset!" << std::endl;
      return false;
    }
    std::vector<float> picture(buffer.cbegin(), buffer.cend());
    if (bias_trick) {
      picture.emplace_back(1);  // Bias trick
    }
    images_.push_back(std::move(picture));
  }
  file.close();

  shuffler.setNelem(images_.size());

  const_images_ = images_;

  return true;
}

void CIFAR10Reader::computeMeanImage() {
  if (!datasetIsLoaded()) {
    return;
  }
  std::cout << "Computing mean image" << std::endl;
  // Compute mean image
  mean_image_.resize(const_images_.front().size(), 0.f);

  const auto num_images = static_cast<float>(const_images_.size());
  for (const auto& image : const_images_) {
    for (size_t pix = 0; pix < image.size(); ++pix) {
      mean_image_[pix] += image.at(pix) / num_images;
    }
  }
}

void CIFAR10Reader::computeStdImage() {
  if (!datasetIsLoaded()) {
    return;
  }
  // Check mean image
  if (!meanImageIsComputed()) {
    computeMeanImage();
  }
  std::cout << "Computing std image" << std::endl;
  std_image_.resize(const_images_.front().size(), 0.f);
  for (const auto& image : const_images_) {
    for (size_t pix = 0; pix < image.size(); ++pix) {
      const float diff = (image.at(pix) - mean_image_.at(pix));
      std_image_[pix] += diff * diff;
    }
  }
  for (size_t d = 0; d < std_image_.size(); ++d) {
    std_image_[d] = sqrtf(std_image_[d] / const_images_.size());
  }
}

bool CIFAR10Reader::normalize() {
  if (!datasetIsLoaded()) {
    return false;
  }
  if (datasetWasPreprocessed()) {
    return false;
  }
  auto minmax = this->minmax();
  float range = minmax.second - minmax.first;
  for (size_t i = 0; i < images_.size(); ++i) {
    std::vector<float>* image = &images_[i];
    for (size_t d = 0; d < image->size(); ++d) {
      image->at(d) = (image->at(d)) / range;
    }
  }
  state_ = PreprocessingType::NORMALIZED;
  return true;
}

bool CIFAR10Reader::standardize() {
  if (!datasetIsLoaded()) {
    return false;
  }
  if (datasetWasPreprocessed()) {
    return false;
  }
  if (!meanImageIsComputed()) {
    computeMeanImage();
  }
  if (!stdImageIsComputed()) {
    computeStdImage();
  }
  float max = std::numeric_limits<float>::min();
  float min = std::numeric_limits<float>::max();
  for (size_t i = 0; i < images_.size(); ++i) {
    for (size_t d = 0; d < images_.front().size(); ++d) {
      images_[i][d] -= mean_image_[d];
      images_[i][d] /= std_image_[d] + 0.0001f;
      if (max < images_[i][d]) max = images_[i][d];
      if (min > images_[i][d]) min = images_[i][d];
    }
  }
  std::cout << "Max: " << max << " min: " << min << std::endl;
  state_ = PreprocessingType::STANDARDIZED;
  return true;
}

bool CIFAR10Reader::demean() {
  if (!datasetIsLoaded()) {
    return false;
  }
  if (datasetWasPreprocessed()) {
    return false;
  }
  if (!meanImageIsComputed()) {
    computeMeanImage();
  }
  // Demean every image
  for (auto& image : images_) {
    for (size_t d = 0; d < image.size(); ++d) {
      image.at(d) -= mean_image_[d];
    }
  }

  // Print some helpful info into console
  std::cout << "Subtracted mean from every image." << std::endl;
  auto minmax = std::minmax_element(mean_image_.begin(), mean_image_.end());
  std::cout << "Mean image min " << *minmax.first << " max " << *minmax.second
            << std::endl;
  state_ = PreprocessingType::DEMEANED;
  return true;
}

bool CIFAR10Reader::reset() {
  images_ = const_images_;
  mean_image_.clear();
  std_image_.clear();
  state_ = PreprocessingType::NO_PREPROCESSING;
  return true;
}

bool CIFAR10Reader::getImage(const size_t index, Image& image) const {
  if (datasetIsLoaded() && index < const_images_.size()) {
    image = Image(const_images_.at(index), KNCols, KNRows, kNumOfChannels);
    return true;
  }
  return false;
}

bool CIFAR10Reader::meanImageIsComputed() const {
  return !mean_image_.empty() && images_.size() > 1 &&
         mean_image_.size() == images_.front().size();
}

bool CIFAR10Reader::stdImageIsComputed() const {
  return !std_image_.empty() && images_.size() > 1 &&
         std_image_.size() == images_.front().size();
}

bool CIFAR10Reader::datasetWasPreprocessed() const {
  if (state_ != PreprocessingType::NO_PREPROCESSING) {
    std::cerr << "The images have already been preprocessed, please undo "
                 "previous preprocessing "
                 "via clicking reset before proceeding."
              << std::endl;
    return true;
  }
  return false;
}

bool CIFAR10Reader::datasetIsLoaded() const {
  return !images_.empty() && !const_images_.empty() &&
         images_.size() == const_images_.size();
}

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

#include <classifiers/mainwindow.h>
#include "ui_mainwindow.h"

namespace {
const std::vector<std::string> kCIFAR10Labels = {"plane", "car",  "bird",  "cat",  "deer",
                                                 "dog",   "frog", "horse", "ship", "truck"};

void weight2image(const CMatrix<float>& w, const size_t label, QImage& img) {
  for (int x = 0; x < 32; ++x) {
    for (int y = 0; y < 32; ++y) {
      int red = w(label, y * 32 + x);
      int green = w(label, 1024 + y * 32 + x);
      int blue = w(label, 2048 + y * 32 + x);
      img.setPixel(x, y, qRgb(red, green, blue));
    }
  }
}
}  // namespace

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::MainWindow), classifier(new LinearSVM(10, 3073)) {
  ui->setupUi(this);

  classifier->initializeW();

  // Ui init
  // Learning rate
  ui->learningRateBox->setRange(0, 22);
  ui->learningRateBox->setSingleStep(1);
  ui->learningRateBox->setValue(static_cast<int>(abs(log10(classifier->learning_rate_))));

  // Epochs
  ui->iterBox->setRange(1, 999);
  ui->iterBox->setValue(1);

  // Batch size
  ui->bsBox->setRange(1, 50000);
  ui->bsBox->setValue(100);

  // Regularization
  ui->regBox->setDecimals(22);
  ui->regBox->setSingleStep(0.00001);
  ui->regBox->setValue(static_cast<double>(classifier->lambda_));

  // Gather ui elements into a vec
  // \todo a better way of gathering all elements together?
  ui_labels_score_.push_back(ui->labelPlaneScore);
  ui_labels_score_.push_back(ui->labelCarScore);
  ui_labels_score_.push_back(ui->labelBirdScore);
  ui_labels_score_.push_back(ui->labelCatScore);
  ui_labels_score_.push_back(ui->labelDeerScore);
  ui_labels_score_.push_back(ui->labelDogScore);
  ui_labels_score_.push_back(ui->labelFrogScore);
  ui_labels_score_.push_back(ui->labelHorseScore);
  ui_labels_score_.push_back(ui->labelShipScore);
  ui_labels_score_.push_back(ui->labelTruckScore);

  ui_labels_loss_.push_back(ui->labelPlaneLoss);
  ui_labels_loss_.push_back(ui->labelCarLoss);
  ui_labels_loss_.push_back(ui->labelBirdLoss);
  ui_labels_loss_.push_back(ui->labelCatLoss);
  ui_labels_loss_.push_back(ui->labelDeerLoss);
  ui_labels_loss_.push_back(ui->labelDogLoss);
  ui_labels_loss_.push_back(ui->labelFrogLoss);
  ui_labels_loss_.push_back(ui->labelHorseLoss);
  ui_labels_loss_.push_back(ui->labelShipLoss);
  ui_labels_loss_.push_back(ui->labelTruckLoss);

  ui_weight_labels_.push_back(ui->w1label);
  ui_weight_labels_.push_back(ui->w2label);
  ui_weight_labels_.push_back(ui->w3label);
  ui_weight_labels_.push_back(ui->w4label);
  ui_weight_labels_.push_back(ui->w5label);
  ui_weight_labels_.push_back(ui->w6label);
  ui_weight_labels_.push_back(ui->w7label);
  ui_weight_labels_.push_back(ui->w8label);
  ui_weight_labels_.push_back(ui->w9label);
  ui_weight_labels_.push_back(ui->w10label);

  resetUI();
}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::updateImage() {
  const size_t index = static_cast<size_t>(ui->labelSpinBox->value());
  if (index >= trainset.images().size()) return;
  const auto cifar_image = trainset.getImage(index);
  QImage img(cifar_image.width(), cifar_image.height(), QImage::Format_RGB888);
  for (size_t x = 0; x < cifar_image.width(); ++x) {
    for (size_t y = 0; y < cifar_image.height(); ++y) {
      int red = cifar_image(x, y, Image::Channel::RED);
      int green = cifar_image(x, y, Image::Channel::GREEN);
      int blue = cifar_image(x, y, Image::Channel::BLUE);
      img.setPixel(static_cast<int>(x), static_cast<int>(y), qRgb(red, green, blue));
    }
  }
  img = img.scaled(ui->piclabel->width(), ui->piclabel->height(), Qt::KeepAspectRatio);
  ui->piclabel->setPixmap(QPixmap::fromImage(img));
  img =
      img.scaled(ui->piclabel_zoomed->width(), ui->piclabel_zoomed->height(), Qt::KeepAspectRatio);
  ui->piclabel_zoomed->setPixmap(QPixmap::fromImage(img));
  ui->labelLineEdit->setText(QString::fromStdString(kCIFAR10Labels[trainset.labels()[index]]));

  std::vector<float> scores = classifier->computeScores(trainset.images()[index]);
  int predicted_label = classifier->infer(trainset.images()[index]);
  ui->predictionLineEdit->setText(QString::fromStdString(kCIFAR10Labels[predicted_label]));

  for (size_t i = 0; i < ui_labels_score_.size(); ++i) {
    ui_labels_score_[i]->setText(QString::number(static_cast<double>(scores[i]), 10, 5));
  }

  // Evaluate loss vector
  std::vector<float> loss_vec =
      classifier->inferenceLoss(trainset.images()[index], trainset.labels()[index]);
  for (size_t i = 0; i < ui_labels_loss_.size(); ++i) {
    ui_labels_loss_[i]->setText(QString::number(static_cast<double>(loss_vec[i]), 10, 5));
  }
}

void MainWindow::on_actionOpen_dataset_triggered() {
  QString folder_path = QFileDialog::getExistingDirectory(this, tr("Load CIFAR dataset"), "");
  if (folder_path.isEmpty()) return;

  QDir dir(folder_path);

  if (!dir.exists()) {
    return;
  }

  dir.setFilter(QDir::Files | QDir::Hidden | QDir::NoSymLinks);
  QStringList filters;
  filters << "data_batch_1.bin";
  filters << "data_batch_2.bin";
  filters << "data_batch_3.bin";
  filters << "data_batch_4.bin";
  filters << "data_batch_5.bin";
  dir.setNameFilters(filters);

  foreach (QFileInfo mitm, dir.entryInfoList()) {
    trainset.readBin(mitm.absoluteFilePath().toUtf8().constData(), true);
  }

  filters.clear();
  filters << "test_batch.bin";
  dir.setNameFilters(filters);
  foreach (QFileInfo mitm, dir.entryInfoList()) {
    testset.readBin(mitm.absoluteFilePath().toUtf8().constData(), true);
  }

  updateImage();
}

void MainWindow::visualizeWeights() {
  CMatrix<float> normW = classifier->W_;
  normW.normalize(0, 255);
  for (size_t i = 0; i < ui_weight_labels_.size(); ++i) {
    auto& weight_label = ui_weight_labels_[i];
    // \todo remove hard coded size
    QImage img(32, 32, QImage::Format_RGB888);
    weight2image(normW, i, img);
    img = img.scaledToWidth(weight_label->width(), Qt::SmoothTransformation);
    weight_label->setPixmap(QPixmap::fromImage(img));
    weight_label->show();
  }
}

float MainWindow::evaluateAcc() {
  int correct = 0;
  int total = 0;
  for (size_t i = 0; i < testset.images().size(); ++i) {
    int label = classifier->infer(testset.images()[i]);
    if (label == testset.labels()[i]) correct++;
    total++;
  }

  return correct / static_cast<float>(total);
}

void MainWindow::on_pushButton_clicked() {
  if (trainset.images().empty()) {
    std::cerr << "Training set is empty! Unable to start training." << std::endl;
  }

  stopped_ = false;
  int bs = ui->bsBox->value();
  int iters = static_cast<int>(trainset.images().size()) / bs;
  for (int epoch = 0; epoch < ui->iterBox->value(); epoch++) {
    for (int i = 0; i < iters; ++i) {
      float loss =
          classifier->computeLoss(trainset.images(), trainset.labels(), trainset.getBatchIdxs(bs));

      ui->lossLabel->setText("Loss: " + QString::number(loss));
      visualizeWeights();
      updateImage();
      qApp->processEvents();
      if (stopped_) return;
      float Wmax = classifier->W_.max();
      float Wmin = classifier->W_.min();
      ui->labelWMax->setText("WMax: " + QString::number(Wmax));
      ui->labelWMin->setText("WMin: " + QString::number(Wmin));

      float dWmax = classifier->dW_.max();
      float dWmin = classifier->dW_.min();
      ui->labeldWMax->setText("dWMax: " + QString::number(dWmax));
      ui->labeldWMin->setText("dWMin: " + QString::number(dWmin));

      ui->labelUpdMax->setText("UpdMax: " + QString::number(dWmax * classifier->learning_rate_));
      ui->labelUpdMin->setText("UpdMin: " + QString::number(dWmin * classifier->learning_rate_));

      ui->labelRatio->setText("Ratio: " + QString::number(classifier->computeWeightRatio()));
    }

    float acc = evaluateAcc();
    std::cout << "Accuracy " << acc << std::endl;
    ui->accLabel->setText("Accuracy: " + QString::number(acc));
  }
}

void MainWindow::on_labelSpinBox_valueChanged(int arg1) { updateImage(); }

void MainWindow::on_stopButton_clicked() { stopped_ = true; }

void MainWindow::on_resetButton_clicked() {
  classifier->initializeW();
  visualizeWeights();
}

void MainWindow::on_learningRateBox_valueChanged(int lr_exp) {
  classifier->learning_rate_ = std::pow(10.f, -ui->learningRateBox->value());
  std::cout << "New learning rate value " << std::pow(10.f, -ui->learningRateBox->value())
            << std::endl;
}

void MainWindow::on_SVMRadioButton_clicked() {
  gW = classifier->W_;
  classifier.reset(new LinearSVM(10, 3073));
  classifier->copyW(gW);
  visualizeWeights();
}

void MainWindow::on_SoftmaxRadioButton_clicked() {
  gW = classifier->W_;
  classifier.reset(new LinearSoftmax(10, 3073));
  classifier->copyW(gW);
  visualizeWeights();
}

void MainWindow::on_regBox_valueChanged(double regularizer) {
  classifier->lambda_ = regularizer;
  std::cout << "New regularization value " << regularizer << std::endl;
}

void MainWindow::on_buttonMeanImage_clicked() {
  // Demean test set
  if (!trainset.demean()) {
    std::cerr << "Unable to demean training set" << std::endl;
    return;
  }
  // Demean training set
  testset.setMeanImage(trainset.meanImage());
  if (!testset.demean()) {
    std::cerr << "Unable to demean test set" << std::endl;
    return;
  }

  // Show mean image
  QImage img(32, 32, QImage::Format_RGB888);
  for (int x = 0; x < 32; ++x) {
    for (int y = 0; y < 32; ++y) {
      int red = trainset.meanImage()[y * 32 + x];
      int green = trainset.meanImage()[1024 + y * 32 + x];
      int blue = trainset.meanImage()[2048 + y * 32 + x];
      img.setPixel(x, y, qRgb(red, green, blue));
    }
  }

  img = img.scaled(ui->labelMeanImage->width(), ui->labelMeanImage->height(), Qt::KeepAspectRatio);
  ui->labelMeanImage->setPixmap(QPixmap::fromImage(img));

  auto minmax = trainset.minmax();
  ui->dataMin->setText("Min: " + QString::number(minmax.first));
  ui->dataMax->setText("Max: " + QString::number(minmax.second));
  ui->labelNormalizationStatus->setText("Images are: demeaned");
}

void MainWindow::on_buttonNormalizationReset_clicked() {
  trainset.reset();
  testset.reset();
  auto minmax = trainset.minmax();
  ui->dataMin->setText("Min: " + QString::number(minmax.first));
  ui->dataMax->setText("Max: " + QString::number(minmax.second));

  /// \todo also reset picture in gui
  ui->labelMeanImage->clear();
  ui->stdImage->clear();
  ui->labelNormalizationStatus->setText("Images are: pristine");
}

void MainWindow::on_buttonStandardize_clicked() {
  // Standardize trainset
  if (!trainset.standardize()) {
    std::cerr << "Unable to standardize training set" << std::endl;
    return;
  }

  // Standardize testset
  testset.setMeanImage(trainset.meanImage());
  testset.setStdImage(trainset.stdImage());
  if (!testset.standardize()) {
    std::cerr << "Unable to standardize test set" << std::endl;
    return;
  }

  // Show mean image
  QImage img(32, 32, QImage::Format_RGB888);
  for (int x = 0; x < 32; ++x) {
    for (int y = 0; y < 32; ++y) {
      int red = trainset.meanImage()[y * 32 + x];
      int green = trainset.meanImage()[1024 + y * 32 + x];
      int blue = trainset.meanImage()[2048 + y * 32 + x];
      img.setPixel(x, y, qRgb(red, green, blue));
    }
  }

  img = img.scaled(ui->labelMeanImage->width(), ui->labelMeanImage->height(), Qt::KeepAspectRatio);
  ui->labelMeanImage->setPixmap(QPixmap::fromImage(img));

  // Show std image
  QImage img2(32, 32, QImage::Format_RGB888);
  for (int x = 0; x < 32; ++x) {
    for (int y = 0; y < 32; ++y) {
      int red = trainset.stdImage()[y * 32 + x];
      int green = trainset.stdImage()[1024 + y * 32 + x];
      int blue = trainset.stdImage()[2048 + y * 32 + x];
      img2.setPixel(x, y, qRgb(red, green, blue));
    }
  }

  img2 = img2.scaled(ui->stdImage->width(), ui->stdImage->height(), Qt::KeepAspectRatio);
  ui->stdImage->setPixmap(QPixmap::fromImage(img2));

  auto minmax = trainset.minmax();
  ui->dataMin->setText("Min: " + QString::number(minmax.first));
  ui->dataMax->setText("Max: " + QString::number(minmax.second));
  ui->labelNormalizationStatus->setText("Images are: standardized");
}

void MainWindow::on_buttonNormalize_clicked() {
  std::cerr << "WARNING: normalization isn't implemented correctly now!" << std::endl;
  if (!trainset.normalize()) {
    std::cerr << "Unable to normalize training set" << std::endl;
    return;
  }
  /// \todo should the normalization be the same??
  if (!testset.normalize()) {
    std::cerr << "Unable to normalize test set" << std::endl;
    return;
  }
  const auto minmax = trainset.minmax();
  ui->dataMin->setText("Min: " + QString::number(minmax.first));
  ui->dataMax->setText("Max: " + QString::number(minmax.second));
  ui->labelNormalizationStatus->setText("Images are: normalized");
}

void MainWindow::resetUI() {
  ui->labelMeanImage->clear();
  ui->stdImage->clear();
  visualizeWeights();
  ///\todo all images...
}

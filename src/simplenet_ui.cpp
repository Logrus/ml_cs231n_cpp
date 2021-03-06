#include <classifiers/simplenet_ui.h>
#include "ui_simplenet_ui.h"

SimpleNetUI::SimpleNetUI(QWidget* parent)
    : QMainWindow(parent),
      ui(new Ui::SimpleNetUI),
      classifier(new SimpleNeuralNet(3072, 70, 10, 0.0001)) {
  ui->setupUi(this);

  classifier->initializeW();

  // TODO: remove later
  label_names.emplace_back("plane");
  label_names.emplace_back("car");
  label_names.emplace_back("bird");
  label_names.emplace_back("cat");
  label_names.emplace_back("deer");
  label_names.emplace_back("dog");
  label_names.emplace_back("frog");
  label_names.emplace_back("horse");
  label_names.emplace_back("ship");
  label_names.emplace_back("truck");

  //  QDir dir("../CIFAR10/");

  //  if (!dir.exists()) {
  //    return;
  //  }

  //  dir.setFilter(QDir::Files | QDir::Hidden | QDir::NoSymLinks);
  //  QStringList filters;
  //  filters << "data_batch_1.bin";
  //  filters << "data_batch_2.bin";
  //  filters << "data_batch_3.bin";
  //  filters << "data_batch_4.bin";
  //  filters << "data_batch_5.bin";
  //  dir.setNameFilters(filters);

  //  foreach (QFileInfo mitm, dir.entryInfoList()) {
  //    trainset.read_bin(mitm.absoluteFilePath().toUtf8().constData(), false);
  //  }

  //  filters.clear();
  //  filters << "test_batch.bin";
  //  dir.setNameFilters(filters);
  //  foreach (QFileInfo mitm, dir.entryInfoList()) {
  //    testset.read_bin(mitm.absoluteFilePath().toUtf8().constData(), false);
  //  }

  // Ui init
  // Learning rate
  ui->learningRateBox->setRange(0, 22);
  ui->learningRateBox->setSingleStep(1);
  ui->learningRateBox->setValue(abs(log10(classifier->learning_rate)));

  // Epochs
  ui->iterBox->setRange(1, 999);
  ui->iterBox->setValue(1);

  // Batch size
  ui->bsBox->setRange(1, 50000);
  ui->bsBox->setValue(100);

  // Regularization
  ui->regBox->setDecimals(22);
  ui->regBox->setSingleStep(0.00001);
  ui->regBox->setValue(classifier->lambda);

  // updateImage();

  TableWidget = new QTableWidget(this);
  TableWidget->setRowCount(10);
  TableWidget->setColumnCount(7);
  TableWidget->horizontalHeader()->setDefaultSectionSize(100);
  TableWidget->verticalHeader()->setDefaultSectionSize(100);
#ifdef Qt4_FOUND
  TableWidget->horizontalHeader()->setResizeMode(QHeaderView::Fixed);
  TableWidget->verticalHeader()->setResizeMode(QHeaderView::Fixed);
#else
  TableWidget->horizontalHeader()->setSectionResizeMode(QHeaderView::Fixed);
  TableWidget->verticalHeader()->setSectionResizeMode(QHeaderView::Fixed);
#endif
  TableWidget->resize(801, 321);
  TableWidget->move(10, 400);

  QPalette pal;
  pal.setColor(QPalette::Background, Qt::red);

  ui->groupBox_2->setPalette(pal);

  if (classifier->loadWeights("weights.dat")) {
    float acc = evaluateAcc();
    std::cout << "Accuracy " << acc << std::endl;
    ui->accLabel->setText("Accuracy: " + QString::number(acc));
  }
  visualizeWeights();

  // Initialize plotting
  // generate some data:
  QVector<double> x(101), y(101);  // initialize with entries 0..100
  for (int i = 0; i < 101; ++i) {
    x[i] = i / 50.0 - 1;  // x goes from -1 to 1
    y[i] = x[i] * x[i];   // let's plot a quadratic function
  }
  // create graph and assign data to it:

  ui->lossplot->addGraph();
  // ui->lossplot->graph(0)->setData(x, y);
  // give the axes some labels:
  ui->lossplot->xAxis->setLabel("Iteration");
  ui->lossplot->yAxis->setLabel("Loss");
  //    // set axes ranges, so we see all data:
  //    ui->lossplot->xAxis->setRange(-1, 1);
  //    ui->lossplot->yAxis->setRange(0, 10);
  ui->lossplot->replot();

  ui->accplot->addGraph();
  ui->accplot->graph(0)->setPen(QPen(Qt::green));
  ui->accplot->graph(0)->setName("val");

  ui->accplot->addGraph();
  ui->accplot->graph(1)->setPen(QPen(Qt::blue));
  ui->accplot->graph(1)->setName("train");
  // ui->accplot->graph(0)->setData(x, y);
  // give the axes some labels:
  ui->accplot->xAxis->setLabel("Epoch");
  ui->accplot->yAxis->setLabel("Accuracy");
  //    // set axes ranges, so we see all data:
  //    ui->accplot->xAxis->setRange(-1, 1);
  //    ui->accplot->yAxis->setRange(0, 10);
  ui->accplot->replot();
}

SimpleNetUI::~SimpleNetUI() {
  delete ui;
  delete classifier;
}

void SimpleNetUI::updateImage() {
  int index = ui->labelSpinBox->value();
  auto mx = trainset.minmax();
  float r = mx.second - mx.first;
  QImage img(32, 32, QImage::Format_RGB888);
  for (int x = 0; x < 32; ++x) {
    for (int y = 0; y < 32; ++y) {
      int red = (trainset.images()[index][y * 32 + x] - mx.first) / r * 255.0;
      int green = (trainset.images()[index][1024 + y * 32 + x] - mx.first) / r * 255.0;
      int blue = (trainset.images()[index][2048 + y * 32 + x] - mx.first) / r * 255.0;
      img.setPixel(x, y, qRgb(red, green, blue));
    }
  }
  img = img.scaled(ui->piclabel->width(), ui->piclabel->height(), Qt::KeepAspectRatio);
  ui->piclabel->setPixmap(QPixmap::fromImage(img));
  ui->labelLineEdit->setText(QString::fromStdString(label_names[trainset.labels()[index]]));

  int predicted_label = classifier->inference(trainset.images()[index]);
  ui->predictionLineEdit->setText(QString::fromStdString(label_names[predicted_label]));

  auto hid = classifier->H_;
  auto minmax = std::minmax_element(hid.begin(), hid.end());
  float range = *minmax.second - *minmax.first;
  QImage hidden(70, 1, QImage::Format_RGB888);
  for (int x = 0; x < 70; ++x) {
    int val = (hid[x] - *minmax.first) / range * 255.0;
    int red = val;
    int green = val;
    int blue = val;
    hidden.setPixel(x, 0, qRgb(red, green, blue));
  }
  hidden = hidden.scaledToWidth(ui->pic_hidden->width(), Qt::FastTransformation);
  ui->pic_hidden->setPixmap(QPixmap::fromImage(hidden));

  auto scores = classifier->inference_scores(trainset.images()[index]);
  minmax = std::minmax_element(scores.begin(), scores.end());
  range = *minmax.second - *minmax.first;
  QImage sores_pic(10, 1, QImage::Format_RGB888);
  for (int x = 0; x < 10; ++x) {
    int val = (scores[x] - *minmax.first) / range * 255.0;
    int red = val;
    int green = val;
    int blue = val;
    sores_pic.setPixel(x, 0, qRgb(red, green, blue));
  }
  sores_pic = sores_pic.scaledToWidth(ui->pic_softmax->width(), Qt::FastTransformation);
  ui->pic_softmax->setPixmap(QPixmap::fromImage(sores_pic));
}

void SimpleNetUI::on_actionOpen_dataset_triggered() {
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
    trainset.readBin(mitm.absoluteFilePath().toUtf8().constData(), false);
  }

  filters.clear();
  filters << "test_batch.bin";
  dir.setNameFilters(filters);
  foreach (QFileInfo mitm, dir.entryInfoList()) {
    testset.readBin(mitm.absoluteFilePath().toUtf8().constData(), false);
  }

  // updateImage();
}

void weight2image(const CMatrix<float>& w, int label, QImage& img) {
  for (int x = 0; x < 32; ++x) {
    for (int y = 0; y < 32; ++y) {
      int red = w(label, y * 32 + x);
      int green = w(label, 1024 + y * 32 + x);
      int blue = w(label, 2048 + y * 32 + x);
      img.setPixel(x, y, qRgb(red, green, blue));
    }
  }
}

void SimpleNetUI::visualizeWeights() {
  CMatrix<float> normW = classifier->W1;
  normW.normalize(0, 255);

  for (int i = 0; i < 70; i++) {
    int ix = i / 7;
    int iy = i % 7;
    // delete TableWidget->item(ix, iy);
    auto* img = new QImage(32, 32, QImage::Format_RGB888);
    weight2image(normW, i, *img);
    *img = img->scaled(100, 100);

    auto* item = new QTableWidgetItem;
    item->setData(Qt::DecorationRole, QPixmap::fromImage(*img));
    TableWidget->setItem(ix, iy, item);
    delete img;
  }
  // TableWidget->repaint();
  TableWidget->viewport()->update();
}

float SimpleNetUI::evaluateAcc() {
  std::fill(classifier->neural_statistics.begin(), classifier->neural_statistics.end(), 0);
  int correct = 0;
  int total = 0;
  for (int i = 0; i < testset.images().size(); ++i) {
    int label = classifier->inference(testset.images()[i]);
    if (label == testset.labels()[i]) correct++;
    total++;
  }

  int counter = 0;

  for (auto a : classifier->neural_statistics) {
    if (a == 0) counter++;
  }
  std::cout << std::endl;

  std::cout << "Dead neurons " << counter << std::endl;

  return correct / static_cast<float>(total);
}

float SimpleNetUI::evaluateTrainAcc() {
  int correct = 0;
  int total = 0;
  for (int i = 0; i < trainset.images().size(); ++i) {
    int label = classifier->inference(trainset.images()[i]);
    if (label == trainset.labels()[i]) correct++;
    total++;
  }
  return correct / static_cast<float>(total);
}

void SimpleNetUI::on_pushButton_clicked() {
  float maxloss = 0, maxacc = 0;
  float minloss = 1e16, minacc = 101;
  int true_iter = 0;
  int true_epochs = 0;
  stopped_ = false;
  int bs = ui->bsBox->value();
  int iters = trainset.images().size() / bs;
  for (int epoch = 0; epoch < ui->iterBox->value(); epoch++) {
    for (int i = 0; i < iters; ++i) {
      // Experiment with Leaky ReLU
      // if (i >=5 && i<=50){
      //     classifier->learning_rate = 0.01;
      // } else {
      //     classifier->learning_rate = std::pow(10.f,-ui->learningRateBox->value());
      // }

      float loss =
          classifier->loss(trainset.images(), trainset.labels(), trainset.getBatchIdxs(bs));
      std::cout << " Loss " << loss << std::endl;

      maxloss = maxloss > loss ? maxloss : loss;
      minloss = minloss < loss ? minloss : loss;

      ui->lossLabel->setText("Loss: " + QString::number(loss));
      if (ui->updWeightViz->isChecked()) {
        visualizeWeights();
      }
      updateImage();
      qApp->processEvents();
      if (stopped_) return;
      float Wmax = classifier->W1.max();
      float Wmin = classifier->W1.min();
      ui->labelWMax->setText("WMax: " + QString::number(Wmax));
      ui->labelWMin->setText("WMin: " + QString::number(Wmin));

      float dWmax = classifier->dW1.max();
      float dWmin = classifier->dW1.min();
      ui->labeldWMax->setText("dWMax: " + QString::number(dWmax));
      ui->labeldWMin->setText("dWMin: " + QString::number(dWmin));

      ui->labelUpdMax->setText("UpdMax: " + QString::number(dWmax * classifier->learning_rate));
      ui->labelUpdMin->setText("UpdMin: " + QString::number(dWmin * classifier->learning_rate));

      // ui->labelRatio->setText("Ratio: " + QString::number(classifier->weight_ratio()));

      loss_stats.push_back(loss);
      iterations.push_back(true_iter);
      true_iter++;

      ui->lossplot->graph(0)->setData(iterations, loss_stats);
      ui->lossplot->graph(0)->rescaleAxes();
      ui->lossplot->replot();
    }

    float acc = evaluateAcc();
    std::cout << "Accuracy " << acc << std::endl;
    ui->accLabel->setText("Accuracy: " + QString::number(acc));

    float trainacc = evaluateTrainAcc();
    std::cout << "Train Accuracy " << trainacc << std::endl;

    maxacc = maxacc > acc ? maxacc : acc;
    minacc = minacc < acc ? minacc : acc;
    maxacc = maxacc > trainacc ? maxacc : trainacc;
    minacc = minacc < trainacc ? minacc : trainacc;

    val_acc_stats.push_back(acc);
    train_acc_stats.push_back(trainacc);
    epochs.push_back(true_epochs);
    true_epochs++;

    ui->accplot->graph(0)->setData(epochs, val_acc_stats);
    ui->accplot->graph(1)->setData(epochs, train_acc_stats);
    ui->accplot->graph(0)->rescaleAxes();
    ui->accplot->graph(1)->rescaleAxes(true);
    ui->accplot->replot();

    if (classifier->saveWeights("weights.dat")) {
      std::cout << "Weights has been saved." << std::endl;
    }

    std::cout << "b1: ";
    for (auto a : classifier->b1) {
      std::cout << a << " ";
    }
    std::cout << std::endl;

    std::cout << "b2: ";
    for (auto a : classifier->b2) {
      std::cout << a << " ";
    }
    std::cout << std::endl;
  }
}

void SimpleNetUI::on_labelSpinBox_valueChanged(int  /*arg1*/) { updateImage(); }

void SimpleNetUI::on_stopButton_clicked() { stopped_ = true; }

void SimpleNetUI::on_resetButton_clicked() {
  classifier->initializeW();
  visualizeWeights();
  val_acc_stats.clear();
  train_acc_stats.clear();
  loss_stats.clear();
}

void SimpleNetUI::on_learningRateBox_valueChanged(int  /*lr_exp*/) {
  classifier->learning_rate = std::pow(10.f, -ui->learningRateBox->value());
  std::cout << "New learning rate value " << std::pow(10.f, -ui->learningRateBox->value())
            << std::endl;
}

void SimpleNetUI::on_regBox_valueChanged(double regularizer) {
  classifier->lambda = regularizer;
  std::cout << "New regularization value " << regularizer << std::endl;
}

void SimpleNetUI::on_buttonMeanImage_clicked() {
  // Demean test set
  trainset.computeMeanImage();
  trainset.demean();
  // Demean training set
  testset.setMeanImage(trainset.meanImage());
  testset.demean();

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
}

void SimpleNetUI::on_buttonNormalizationReset_clicked() {
  trainset.reset();
  testset.reset();
  auto minmax = trainset.minmax();
  ui->dataMin->setText("Min: " + QString::number(minmax.first));
  ui->dataMax->setText("Max: " + QString::number(minmax.second));
}

void SimpleNetUI::on_buttonStandardize_clicked() {
  // Standardize trainset
  trainset.computeMeanImage();
  trainset.computeStdImage();
  trainset.standardize();
  // Standardize testset
  testset.setMeanImage(trainset.meanImage());
  testset.setStdImage(trainset.stdImage());
  testset.standardize();

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
}

void SimpleNetUI::on_buttonNormalize_clicked() {
  trainset.normalize();
  testset.normalize();
  auto minmax = trainset.minmax();
  ui->dataMin->setText("Min: " + QString::number(minmax.first));
  ui->dataMax->setText("Max: " + QString::number(minmax.second));
}

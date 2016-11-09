#include "simplenet_ui.h"
#include "ui_simplenet_ui.h"

SimpleNetUI::SimpleNetUI(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::SimpleNetUI),
    classifier(new SimpleNeuralNet(3072, 70, 10, 0.0001))
{
    ui->setupUi(this);

    classifier->initializeW();

    // TODO: remove later
    label_names.push_back("plane");
    label_names.push_back("car");
    label_names.push_back("bird");
    label_names.push_back("cat");
    label_names.push_back("deer");
    label_names.push_back("dog");
    label_names.push_back("frog");
    label_names.push_back("horse");
    label_names.push_back("ship");
    label_names.push_back("truck");

    QDir dir("../CIFAR10/");

    if(!dir.exists()) {
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

    foreach(QFileInfo mitm, dir.entryInfoList()){

        trainset.read_bin(mitm.absoluteFilePath().toUtf8().constData(), false);
    }

    filters.clear();
    filters << "test_batch.bin";
    dir.setNameFilters(filters);
    foreach(QFileInfo mitm, dir.entryInfoList()){
        testset.read_bin(mitm.absoluteFilePath().toUtf8().constData(), false);
    }

    // Ui init
    // Learning rate
    ui->learningRateBox->setRange(0,22);
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

    //updateImage();

    //visualizeWeights();
}

SimpleNetUI::~SimpleNetUI()
{
    delete ui;
    delete classifier;
}

void SimpleNetUI::updateImage(){
  int index = ui->labelSpinBox->value();
  QImage img(32, 32, QImage::Format_RGB888);
  for (int x = 0; x < 32; ++x) {
    for (int y = 0; y < 32; ++y) {
        int red=trainset.images_[index][y*32+x];
        int green=trainset.images_[index][1024+y*32+x];
        int blue=trainset.images_[index][2048+y*32+x];
        img.setPixel(x, y, qRgb(red, green, blue));
    }
  }
  img = img.scaled(ui->piclabel->width(), ui->piclabel->height(), Qt::KeepAspectRatio);
  ui->piclabel->setPixmap(QPixmap::fromImage(img));
  ui->labelLineEdit->setText(QString::fromStdString(label_names[trainset.labels_[index]]));

//  std::vector<float> scores = classifier->scores(trainset.images_[index]);
  int predicted_label = classifier->inference(trainset.images_[index]);
  ui->predictionLineEdit->setText(QString::fromStdString( label_names[predicted_label] ));

//  ui->labelPlaneScore->setText(QString::number( scores[0], 10, 5));
//  ui->labelCarScore->setText(QString::number(   scores[1], 10, 5));
//  ui->labelBirdScore->setText(QString::number(  scores[2], 10, 5));
//  ui->labelCatScore->setText(QString::number(   scores[3], 10, 5));
//  ui->labelDeerScore->setText(QString::number(  scores[4], 10, 5));
//  ui->labelDogScore->setText(QString::number(   scores[5], 10, 5));
//  ui->labelFrogScore->setText(QString::number(  scores[6], 10, 5));
//  ui->labelHorseScore->setText(QString::number( scores[7], 10, 5));
//  ui->labelShipScore->setText(QString::number(  scores[8], 10, 5));
//  ui->labelTruckScore->setText(QString::number( scores[9], 10, 5));

//  // Evaluate loss vector
//  std::vector<float> loss_vec = classifier->inference_loss(trainset.images_[index], trainset.labels_[index]);
//  ui->labelPlaneLoss->setText(QString::number( loss_vec[0], 10, 5));
//  ui->labelCarLoss->setText(QString::number(   loss_vec[1], 10, 5));
//  ui->labelBirdLoss->setText(QString::number(  loss_vec[2], 10, 5));
//  ui->labelCatLoss->setText(QString::number(   loss_vec[3], 10, 5));
//  ui->labelDeerLoss->setText(QString::number(  loss_vec[4], 10, 5));
//  ui->labelDogLoss->setText(QString::number(   loss_vec[5], 10, 5));
//  ui->labelFrogLoss->setText(QString::number(  loss_vec[6], 10, 5));
//  ui->labelHorseLoss->setText(QString::number( loss_vec[7], 10, 5));
//  ui->labelShipLoss->setText(QString::number(  loss_vec[8], 10, 5));
//  ui->labelTruckLoss->setText(QString::number( loss_vec[9], 10, 5));

}

void SimpleNetUI::on_actionOpen_dataset_triggered()
{
    QString folder_path = QFileDialog::getExistingDirectory(this, tr("Load CIFAR dataset"), "");
    if(folder_path.isEmpty()) return;

    QDir dir(folder_path);

    if(!dir.exists()) {
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

    foreach(QFileInfo mitm, dir.entryInfoList()){

        trainset.read_bin(mitm.absoluteFilePath().toUtf8().constData(), false);
    }

    filters.clear();
    filters << "test_batch.bin";
    dir.setNameFilters(filters);
    foreach(QFileInfo mitm, dir.entryInfoList()){
        testset.read_bin(mitm.absoluteFilePath().toUtf8().constData(), false);
    }

    //updateImage();
}

void weight2image(CMatrix<float> w, int label, QImage &img){


    for (int x = 0; x < 32; ++x) {
       for (int y = 0; y < 32; ++y) {
         int red=w(label,y*32+x);
         int green=w(label,1024+y*32+x);
         int blue=w(label,2048+y*32+x);
         img.setPixel(x, y, qRgb(red, green, blue));
       }
     }

   }

void SimpleNetUI::visualizeWeights(){
    QImage img0(32, 32, QImage::Format_RGB888);
    QImage img1(32, 32, QImage::Format_RGB888);
    QImage img2(32, 32, QImage::Format_RGB888);
    QImage img3(32, 32, QImage::Format_RGB888);
    QImage img4(32, 32, QImage::Format_RGB888);
    QImage img5(32, 32, QImage::Format_RGB888);
    QImage img6(32, 32, QImage::Format_RGB888);
    QImage img7(32, 32, QImage::Format_RGB888);
    QImage img8(32, 32, QImage::Format_RGB888);
    QImage img9(32, 32, QImage::Format_RGB888);

    CMatrix<float> normW = classifier->W1;
    normW.normalize(0,255);

    weight2image(normW, 0, img0);
    weight2image(normW, 1, img1);
    weight2image(normW, 2, img2);
    weight2image(normW, 3, img3);
    weight2image(normW, 4, img4);
    weight2image(normW, 5, img5);
    weight2image(normW, 6, img6);
    weight2image(normW, 7, img7);
    weight2image(normW, 8, img8);
    weight2image(normW, 9, img9);

    img0=img0.scaledToWidth(ui->w1label->width(), Qt::SmoothTransformation);
    img1=img1.scaledToWidth(ui->w2label->width(), Qt::SmoothTransformation);
    img2=img2.scaledToWidth(ui->w3label->width(), Qt::SmoothTransformation);
    img3=img3.scaledToWidth(ui->w4label->width(), Qt::SmoothTransformation);
    img4=img4.scaledToWidth(ui->w5label->width(), Qt::SmoothTransformation);
    img5=img5.scaledToWidth(ui->w6label->width(), Qt::SmoothTransformation);
    img6=img6.scaledToWidth(ui->w7label->width(), Qt::SmoothTransformation);
    img7=img7.scaledToWidth(ui->w8label->width(), Qt::SmoothTransformation);
    img8=img8.scaledToWidth(ui->w9label->width(), Qt::SmoothTransformation);
    img9=img9.scaledToWidth(ui->w10label->width(), Qt::SmoothTransformation);

    ui->w1label->setPixmap(QPixmap::fromImage(img0));
    ui->w2label->setPixmap(QPixmap::fromImage(img1));
    ui->w3label->setPixmap(QPixmap::fromImage(img2));
    ui->w4label->setPixmap(QPixmap::fromImage(img3));
    ui->w5label->setPixmap(QPixmap::fromImage(img4));
    ui->w6label->setPixmap(QPixmap::fromImage(img5));
    ui->w7label->setPixmap(QPixmap::fromImage(img6));
    ui->w8label->setPixmap(QPixmap::fromImage(img7));
    ui->w9label->setPixmap(QPixmap::fromImage(img8));
    ui->w10label->setPixmap(QPixmap::fromImage(img9));

    ui->w1label->show();
    ui->w2label->show();
    ui->w3label->show();
    ui->w4label->show();
    ui->w5label->show();
    ui->w6label->show();
    ui->w7label->show();
    ui->w8label->show();
    ui->w9label->show();
    ui->w10label->show();

}

float SimpleNetUI::evaluateAcc(){
    int correct = 0;
    int total = 0;
    for(int i=0; i<testset.images_.size(); ++i){
       int label = classifier->inference(testset.images_[i]);
       if(label == testset.labels_[i]) correct++;
       total++;
    }

    return correct/static_cast<float>(total);
}

void SimpleNetUI::on_pushButton_clicked()
{
    stopped_ = false;
    int bs = ui->bsBox->value();
    int iters = trainset.images_.size()/bs;
    for(int epoch = 0; epoch < ui->iterBox->value(); epoch++){

        for(int i=0; i<iters; ++i){
            float loss = classifier->loss(trainset.images_, trainset.labels_, trainset.get_batch_idxs(bs));
            std::cout << " Loss " << loss << std::endl;

            ui->lossLabel->setText("Loss: " + QString::number(loss));
            //visualizeWeights();
            //updateImage();
            qApp->processEvents();
            if(stopped_) return;
            float Wmax = classifier->W1.max();
            float Wmin = classifier->W1.min();
            ui->labelWMax->setText("WMax: " + QString::number(Wmax));
            ui->labelWMin->setText("WMin: " + QString::number(Wmin));

            float dWmax = classifier->dW1.max();
            float dWmin = classifier->dW1.min();
            ui->labeldWMax->setText("dWMax: " + QString::number(dWmax));
            ui->labeldWMin->setText("dWMin: " + QString::number(dWmin));

            ui->labelUpdMax->setText("UpdMax: " + QString::number(dWmax*classifier->learning_rate));
            ui->labelUpdMin->setText("UpdMin: " + QString::number(dWmin*classifier->learning_rate));

            //ui->labelRatio->setText("Ratio: " + QString::number(classifier->weight_ratio()));

        }

        float acc = evaluateAcc();
        std::cout << "Accuracy " << acc << std::endl;
        ui->accLabel->setText("Accuracy: " + QString::number(acc));

    }
}

void SimpleNetUI::on_labelSpinBox_valueChanged(int arg1)
{
    updateImage();
}

void SimpleNetUI::on_stopButton_clicked()
{
    stopped_ = true;
}

void SimpleNetUI::on_resetButton_clicked()
{
    classifier->initializeW();
    //visualizeWeights();
}

void SimpleNetUI::on_learningRateBox_valueChanged(int lr_exp)
{
    classifier->learning_rate = std::pow(10.f,-ui->learningRateBox->value());
    std::cout << "New learning rate value " << std::pow(10.f,-ui->learningRateBox->value()) << std::endl;
}

void SimpleNetUI::on_SVMRadioButton_clicked()
{
//    gW = classifier->W;
//    delete classifier;
//    classifier = new LinearSVM(10, 3073);
//    classifier->copyW(gW);
//    visualizeWeights();
}

void SimpleNetUI::on_SoftmaxRadioButton_clicked()
{
//    gW = classifier->W;
//    delete classifier;
//    classifier = new LinearSoftmax(10, 3073);
//    classifier->copyW(gW);
//    visualizeWeights();
}

void SimpleNetUI::on_regBox_valueChanged(double regularizer)
{
    classifier->lambda = regularizer;
    std::cout << "New regularization value " << regularizer << std::endl;
}

void SimpleNetUI::on_buttonMeanImage_clicked()
{
    // Demean test set
    trainset.compute_mean();
    trainset.demean();
    // Demean training set
    testset.mean_image = trainset.mean_image;
    testset.demean();

    // Show mean image
    QImage img(32, 32, QImage::Format_RGB888);
    for (int x = 0; x < 32; ++x) {
      for (int y = 0; y < 32; ++y) {
          int red=trainset.mean_image[y*32+x];
          int green=trainset.mean_image[1024+y*32+x];
          int blue=trainset.mean_image[2048+y*32+x];
          img.setPixel(x, y, qRgb(red, green, blue));
      }
    }

    img = img.scaled(ui->labelMeanImage->width(), ui->labelMeanImage->height(), Qt::KeepAspectRatio);
    ui->labelMeanImage->setPixmap(QPixmap::fromImage(img));

    auto minmax = trainset.minmax();
    ui->dataMin->setText("Min: " + QString::number( minmax.first ));
    ui->dataMax->setText("Max: " + QString::number( minmax.second ));

}

void SimpleNetUI::on_buttonNormalizationReset_clicked()
{
    trainset.reset();
    testset.reset();
    auto minmax = trainset.minmax();
    ui->dataMin->setText("Min: " + QString::number( minmax.first ));
    ui->dataMax->setText("Max: " + QString::number( minmax.second ));
}

void SimpleNetUI::on_buttonStandardize_clicked()
{
    // Standardize trainset
    trainset.compute_mean();
    trainset.compute_std();
    trainset.standardize();
    // Standardize testset
    testset.mean_image = trainset.mean_image;
    testset.std_image = trainset.std_image;
    testset.standardize();

    // Show mean image
    QImage img(32, 32, QImage::Format_RGB888);
    for (int x = 0; x < 32; ++x) {
      for (int y = 0; y < 32; ++y) {
          int red=trainset.mean_image[y*32+x];
          int green=trainset.mean_image[1024+y*32+x];
          int blue=trainset.mean_image[2048+y*32+x];
          img.setPixel(x, y, qRgb(red, green, blue));
      }
    }

    img = img.scaled(ui->labelMeanImage->width(), ui->labelMeanImage->height(), Qt::KeepAspectRatio);
    ui->labelMeanImage->setPixmap(QPixmap::fromImage(img));

    // Show std image
    QImage img2(32, 32, QImage::Format_RGB888);
    for (int x = 0; x < 32; ++x) {
      for (int y = 0; y < 32; ++y) {
          int red=trainset.std_image[y*32+x];
          int green=trainset.std_image[1024+y*32+x];
          int blue=trainset.std_image[2048+y*32+x];
          img2.setPixel(x, y, qRgb(red, green, blue));
      }
    }

    img2 = img2.scaled(ui->stdImage->width(), ui->stdImage->height(), Qt::KeepAspectRatio);
    ui->stdImage->setPixmap(QPixmap::fromImage(img2));


    auto minmax = trainset.minmax();
    ui->dataMin->setText("Min: " + QString::number( minmax.first ));
    ui->dataMax->setText("Max: " + QString::number( minmax.second ));

}

void SimpleNetUI::on_buttonNormalize_clicked()
{
    trainset.normalize();
    testset.normalize();
    auto minmax = trainset.minmax();
    ui->dataMin->setText("Min: " + QString::number( minmax.first ));
    ui->dataMax->setText("Max: " + QString::number( minmax.second ));
}

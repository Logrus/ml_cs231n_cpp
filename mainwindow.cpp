#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    svm(10, 3073)
{
    ui->setupUi(this);

    // TODO: remove later
    label_names.push_back("airplane");
    label_names.push_back("automobile");
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

        reader.read_bin(mitm.absoluteFilePath().toUtf8().constData());
    }
    updateImage();

    visualizeWeights();

    // Ui init
    ui->learningRateBox->setRange(0,1.0e-16);
    ui->learningRateBox->setValue(svm.learning_rate);
    ui->iterBox->setValue(1);

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::updateImage(){
  int index = ui->labelSpinBox->value();
  QImage img(32, 32, QImage::Format_RGB888);
  for (int x = 0; x < 32; ++x) {
    for (int y = 0; y < 32; ++y) {
        int red=reader.images_[index][y*32+x];
        int green=reader.images_[index][1024+y*32+x];
        int blue=reader.images_[index][2048+y*32+x];
        img.setPixel(x, y, qRgb(red, green, blue));
    }
  }
  img = img.scaledToWidth(ui->piclabel->width());
  ui->piclabel->setPixmap(QPixmap::fromImage(img));
  ui->labelLineEdit->setText(QString::fromStdString(label_names[reader.labels_[index]]));

  int predicted_label = svm.inference(reader.images_[index]);
  ui->predictionLineEdit->setText(QString::fromStdString( label_names[reader.labels_[predicted_label]] ));
}

void MainWindow::on_actionOpen_dataset_triggered()
{
    QString folder_path = QFileDialog::getExistingDirectory(this, tr("Load CIFAR dataset"), "");
    if(folder_path.isEmpty()) return;

    QDir dir(folder_path);

    if(!dir.exists()) {
        return;
    }

    dir.setFilter(QDir::Files | QDir::Hidden | QDir::NoSymLinks);
    QStringList filters;
    filters << "*.bin";
    dir.setNameFilters(filters);

    foreach(QFileInfo mitm, dir.entryInfoList()){
        reader.read_bin(mitm.absoluteFilePath().toUtf8().constData());
    }
    updateImage();
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

void MainWindow::visualizeWeights(){
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

    CMatrix<float> normW = svm.W;
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

float MainWindow::evaluateAcc(){
    int correct = 0;
    int total = 0;
    for(int i=5000; i<=6000; ++i){
       int label = svm.inference(reader.images_[i]);
       if(label == reader.labels_[i]) correct++;
       total++;
    }

    return correct/static_cast<float>(total);
}

void MainWindow::on_pushButton_clicked()
{
    stopped_ = false;
    int bs = 100;
    int iters = 50000/bs;
    for(int epoch = 0; epoch < ui->iterBox->value(); epoch++){

        for(int i=0; i<iters; ++i){
            float loss = svm.loss(reader.images_, reader.labels_, i*bs, i*bs + bs);
            ui->lossLabel->setText("Loss: " + QString::number(loss));
            visualizeWeights();
            updateImage();
            qApp->processEvents();
            if(stopped_) return;

        }

        float acc = evaluateAcc();
        std::cout << "Accuracy " << acc << std::endl;
        ui->accLabel->setText("Accuracy: " + QString::number(acc));

    }
}

void MainWindow::on_labelSpinBox_valueChanged(int arg1)
{
    updateImage();
}

void MainWindow::on_stopButton_clicked()
{
    stopped_ = true;
}

void MainWindow::on_resetButton_clicked()
{
    svm.initializeW();
    visualizeWeights();
}

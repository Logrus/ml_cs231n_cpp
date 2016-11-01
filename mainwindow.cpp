#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    svm(10, 3073)
{
    ui->setupUi(this);



    // TODO: remove later

    QDir dir("../CIFAR10/");

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

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::updateImage(){
  int index=0;
  QImage img(32, 32, QImage::Format_RGB888);
  for (int x = 0; x < 32; ++x) {
    for (int y = 0; y < 32; ++y) {
        int red=reader.images_[index][y*32+x];
        int green=reader.images_[index][1024+y*32+x];
        int blue=reader.images_[index][2048+y*32+x];
        img.setPixel(x, y, qRgb(red, green, blue));
    }
  }
  img = img.scaledToWidth(ui->piclabel->width(), Qt::SmoothTransformation);
  ui->piclabel->setPixmap(QPixmap::fromImage(img));
  ui->labelLineEdit->setText(QString::number(reader.labels_[index]));
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

void MainWindow::on_pushButton_clicked()
{
    int bs = 100;
    for(int i=1; i<50; ++i){
        svm.loss(reader.images_, reader.labels_, i*bs, i*bs + bs);
        qApp->processEvents();
    }
}

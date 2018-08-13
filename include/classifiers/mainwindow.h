#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <math.h>
#include <uni_freiburg_cv/CMatrix.h>
#include <QApplication>
#include <QDir>
#include <QFileDialog>
#include <QMainWindow>
#include <iostream>
#include <memory>  // unique_ptr
#include "cifar_reader.h"
#include "classifier.h"
#include "linearsoftmax.h"
#include "linearsvm.h"

class QLabel;
namespace Ui {
class MainWindow;
}  // namespace Ui

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit MainWindow(QWidget* parent = 0);
  ~MainWindow();
  void updateImage();

 private slots:
  void on_actionOpen_dataset_triggered();

  void on_pushButton_clicked();

  void on_labelSpinBox_valueChanged(int arg1);

  void on_stopButton_clicked();

  void on_resetButton_clicked();

  void on_learningRateBox_valueChanged(int arg1);

  void on_SVMRadioButton_clicked();

  void on_SoftmaxRadioButton_clicked();

  void on_regBox_valueChanged(double arg1);

  void on_buttonMeanImage_clicked();

  void on_buttonNormalizationReset_clicked();

  void on_buttonStandardize_clicked();

  void on_buttonNormalize_clicked();

 private:
  void visualizeWeights();
  float evaluateAcc();

  std::vector<std::string> label_names;

  Ui::MainWindow* ui;
  CIFAR10Reader trainset;
  CIFAR10Reader testset;
  std::unique_ptr<Classifier> classifier;

  std::vector<QLabel*> ui_labels_score_;
  std::vector<QLabel*> ui_labels_loss_;

  bool stopped_ = false;

  CMatrix<float> gW;  // Global weights (just for fun!)
};

#endif  // MAINWINDOW_H

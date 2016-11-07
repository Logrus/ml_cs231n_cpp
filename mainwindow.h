#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QApplication>
#include <QMainWindow>
#include <QFileDialog>
#include <QDir>
#include <iostream>
#include "cifar_reader.h"
#include "classifier.h"
#include "linearsvm.h"
#include "linearsoftmax.h"
#include "CMatrix.h"
#include <math.h>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
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

    Ui::MainWindow *ui;
    CIFAR10Reader trainset;
    CIFAR10Reader testset;
    Classifier * classifier;

    bool stopped_ = false;

    CMatrix<float> gW; // Global weights (just for fun!)
};

#endif // MAINWINDOW_H

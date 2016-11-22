#ifndef SIMPLENET_UI_H
#define SIMPLENET_UI_H

#include <QApplication>
#include <QMainWindow>
#include <QFileDialog>
#include <QDir>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <iostream>
#include "cifar_reader.h"
#include "simpleneuralnet.h"
#include "CMatrix.h"
#include <math.h>
#include "qcustomplot.h"

namespace Ui {
class SimpleNetUI;
}

class SimpleNetUI : public QMainWindow
{
    Q_OBJECT

public:
    explicit SimpleNetUI(QWidget *parent = 0);
    ~SimpleNetUI();
    void updateImage();

private slots:
    void on_actionOpen_dataset_triggered();

    void on_pushButton_clicked();

    void on_labelSpinBox_valueChanged(int arg1);

    void on_stopButton_clicked();

    void on_resetButton_clicked();

    void on_learningRateBox_valueChanged(int arg1);

    void on_regBox_valueChanged(double arg1);

    void on_buttonMeanImage_clicked();

    void on_buttonNormalizationReset_clicked();

    void on_buttonStandardize_clicked();

    void on_buttonNormalize_clicked();

private:
    void visualizeWeights();
    float evaluateAcc();
    float evaluateTrainAcc();

    std::vector<std::string> label_names;

    Ui::SimpleNetUI *ui;
    CIFAR10Reader trainset;
    CIFAR10Reader testset;
    SimpleNeuralNet * classifier;
    QTableWidget* TableWidget;

    bool stopped_ = false;

    QVector<double> loss_stats;
    QVector<double> val_acc_stats;
    QVector<double> train_acc_stats;
    QVector<double> iterations;
    QVector<double> epochs;
};


#endif // SIMPLENET_UI_H


#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QApplication>
#include <QMainWindow>
#include <QFileDialog>
#include <QDir>
#include <iostream>
#include "cifar_reader.h"
#include "linearsvm.h"
#include "matrix.h"

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

private:
    void visualizeWeights();
    float evaluateAcc();

    Ui::MainWindow *ui;
    CIFAR10Reader reader;
    LinearSVM svm;
};

#endif // MAINWINDOW_H

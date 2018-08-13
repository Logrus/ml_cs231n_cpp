# ml_cs231n_cpp

This repository contains C++ implementation of the SVM and Softmax loss functions in addition to the simple two-layer Neural Network.
They were developed for visualization and understanding while doing CS231n from Stanford University in 2016.

This project contains:

1. Implementation of the SVM and Softmax loss [video](https://www.youtube.com/watch?v=QrRTKM8xJaA)
2. Implementation of the 2 layer neural network [video](https://www.youtube.com/watch?v=CHpu8N18aRs)
3. Multiple normalization possibilities (mean subtraction, normalization, standardization)
4. Adam weight update
5. Fisher–Yates shuffle algorithm for batching
6. Simple ensemble executable that loads saved weights from the network training and averages at test time

## SVM and Softmax loss
![svmandsoftmaxloss](https://github.com/Logrus/ml_cs231n_cpp/raw/master/images/svm_softmax_viz.png)

## Two layer Neural Network
![twolayernetwork](https://github.com/Logrus/ml_cs231n_cpp/raw/master/images/two_layer_nn_viz.png)

# Dependencies

The project depends on having either Qt4 or Qt5 lib.
It was tested on Ubuntu 18.04 LTS with g++ 7.3.0

# Clone & Build

The build should go as usual with CMake

```
git clone https://github.com/Logrus/ml_cs231n_cpp.git
cd ml_cs231n_cpp
mkdir build
cd build
cmake ..
make
```
# Run
Before you can experiment, you should upload CIFAR10 dataset by running a script:
```
# From ml_cs231n_cpp folder
./get_datasets.sh
```
It will create a `data/CIFAR10` folder in the current folder and download bin files to CIFAR10 folder.

Note: Don't hesitate to change `get_datasets.sh` script if you want to download files to another place.

After compilation you can SVM and Softmax visualizations:
```
# From build folder call
./visualizer
```
In the open window click on `Open dataset` button and specify path to CIFAR10 folder (e.g. ml_cs231n_cpp/data/CIFAR10).

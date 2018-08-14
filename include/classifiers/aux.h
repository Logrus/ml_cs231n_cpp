#pragma once
#include <uni_freiburg_cv/CMatrix.h>

// Gradient check
CMatrix<float> approximate_gradient(float (*lossfun)(const CMatrix<float>&, const int&),
                                    const CMatrix<float> W) {
  CMatrix<float> dW(W.xSize(), W.ySize());

  for (int i = 0; i < dW.size(); ++i) {
  }

  return dW;
}

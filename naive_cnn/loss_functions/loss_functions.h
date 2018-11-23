/*
    Copyright (c) 2018-2019, Lu Feng
    All rights reserved.
*/

#ifndef NAIVE_CNN_LOSS_FUNCTIONS_LOSS_FUNCTIONS_H_
#define NAIVE_CNN_LOSS_FUNCTIONS_LOSS_FUNCTIONS_H_

#include "naive_cnn/util/util.h"

namespace naive_cnn {

// mean-squared-error loss function for regression
class mse {
 public:
  static float_t f(float_t y, float_t t) {
    return (y - t) * (y - t) / 2;
  }

  static float_t df(float_t y, float_t t) {
    return y - t;
  }
};

// cross-entropy loss function for (multiple independent)
// binary classifications
class cross_entropy {
 public:
  static float_t f(float_t y, float_t t) {
    return -t * std::log(y) - (float_t(1) - t) * std::log(float_t(1) - y);
  }

  static float_t df(float_t y, float_t t) {
    return (y - t) / (y * (float_t(1) - y));
  }
};

// cross-entropy loss function for multi-class classification
class cross_entropy_multiclass {
 public:
  static float_t f(float_t y, float_t t) {
    return -t * std::log(y);
  }

  static float_t df(float_t y, float_t t) {
    return - t / y;
  }
};

template <typename E>
vec_t gradient(const vec_t& y, const vec_t& t) {
  vec_t grad(y.size());
  assert(y.size() == t.size());

  for (cnn_size_t i = 0; i < y.size(); i++)
    grad[i] = E::df(y[i], t[i]);

  return grad;
}

}  // namespace naive_cnn

#endif  // NAIVE_CNN_LOSS_FUNCTIONS_LOSS_FUNCTIONS_H_

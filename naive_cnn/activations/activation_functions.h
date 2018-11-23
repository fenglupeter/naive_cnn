/*
    Copyright (c) 2018-2019, Lu Feng
    All rights reserved.
*/

#ifndef NAIVE_CNN_ACTIVATIONS_ACTIVATION_FUNCTIONS_H_
#define NAIVE_CNN_ACTIVATIONS_ACTIVATION_FUNCTIONS_H_

#include <algorithm>
#include <utility>

namespace naive_cnn {

class function {
 public:
  function() = default;
  function(const function &) = default;
#ifndef CNN_DEFAULT_MOVE_CONSTRUCTOR_UNAVAILABLE
  function(function &&) = default;
#endif
  function &operator =(const function &) = default;
#ifndef CNN_DEFAULT_ASSIGNMENT_OPERATOR_UNAVAILABLE
  function &operator =(function &&) = default;
#endif
  virtual ~function() = default;

  virtual float_t f(const vec_t& v, cnn_size_t index) const = 0;

  // dfi/dyi
  virtual float_t df(float_t y) const = 0;

  // dfi/dyk (k=0,1,..n)
  virtual vec_t df(const vec_t& y, cnn_size_t i) const {
    vec_t v(y.size(), 0);
    v[i] = df(y[i]);
    return v;
  }

  // target value range for learning
  virtual std::pair<float_t, float_t> scale() const = 0;
};

class identity : public function {
 public:
  using function::df;
  float_t f(const vec_t& v, cnn_size_t i) const override {
    return v[i];
  }

  float_t df(float_t /*y*/) const override {
    return float_t(1);
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }
};

class sigmoid : public function {
 public:
  using function::df;
  float_t f(const vec_t& v, cnn_size_t i) const override {
    return float_t(1) / (float_t(1) + std::exp(-v[i]));
  }

  float_t df(float_t y) const override {
    return y * (float_t(1) - y);
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }
};

class relu : public function {
 public:
  using function::df;
  float_t f(const vec_t& v, cnn_size_t i) const override {
    return std::max(float_t(0), v[i]);
  }

  float_t df(float_t y) const override {
    return y > float_t(0) ? float_t(1) : float_t(0);
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }
};

// typedef relu rectified_linear;  // for compatibility

class leaky_relu : public function {
 public:
  using function::df;
  float_t f(const vec_t& v, cnn_size_t i) const override {
    return (v[i] > float_t(0)) ? v[i] : float_t(0.01) * v[i];
  }

  float_t df(float_t y) const override {
    return y > float_t(0) ? float_t(1) : float_t(0.01);
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }
};

class elu : public function {
 public:
  using function::df;
  float_t f(const vec_t& v, cnn_size_t i) const override {
    return (v[i] < float_t(0) ? (exp(v[i]) - float_t(1)) : v[i]);
  }

  float_t df(float_t y) const override {
    return (y > float_t(0) ? float_t(1) : (float_t(1)+y));
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }
};

class softmax : public function {
 public:
  float_t f(const vec_t& v, cnn_size_t i) const override {
    float_t alpha = *std::max_element(v.begin(), v.end());
    float_t numer = std::exp(v[i] - alpha);
    float_t denom = float_t(0);
    for (auto x : v)
      denom += std::exp(x - alpha);
    return numer / denom;
  }

  float_t df(float_t y) const override {
    return y * (float_t(1) - y);
  }

  vec_t df(const vec_t& y, cnn_size_t index) const override {
    vec_t v(y.size(), 0);
    for (cnn_size_t i = 0; i < y.size(); i++)
      v[i] = (i == index) ? df(y[index]) : -y[i] * y[index];

    return v;
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0), float_t(1));
  }
};

class tan_h : public function {
 public:
  using function::df;
  float_t f(const vec_t& v, cnn_size_t i) const override {
    const float_t ep = std::exp(v[i]);
    const float_t em = std::exp(-v[i]);
    return (ep - em) / (ep + em);
  }

  // fast approximation of tanh (improve 2-3% speed in LeNet-5)
  //  float_t f(float_t x) const {
  //      const float_t x2 = x * x;
  //      x *= 1.0 + x2 * (0.1653 + x2 * 0.0097);
  //      return x / std::sqrt(1.0 + x * x);
  //      invsqrt(static_cast<float>(1.0 + x * x));
  //  }

  float_t df(float_t y) const override {
    return float_t(1) - sqr(y);
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(-0.8), float_t(0.8));
  }

 private:
  //  float invsqrt(float x) const {
  //      float x2 = x * 0.5f;
  //      long i = *reinterpret_cast<long*>(&x);
  //
  //      i = 0x5f3759df - (i >> 1);
  //      x = *reinterpret_cast<float*>(&i);
  //      x = x * (1.5f - (x2 * x * x));
  //      return x;
  //  }
};

// s tan_h, but scaled to match the other functions
class tan_hp1m2 : public function {
 public:
  using function::df;
  float_t f(const vec_t& v, cnn_size_t i) const override {
    const float_t ep = std::exp(v[i]);
    return ep / (ep + std::exp(-v[i]));
  }

  float_t df(float_t y) const override {
    return 2 * y *(float_t(1) - y);
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }
};

}  // namespace naive_cnn

#endif  // NAIVE_CNN_ACTIVATIONS_ACTIVATION_FUNCTIONS_H_

/*
    Copyright (c) 2018-2019, Lu Feng
    All rights reserved.
*/

#ifndef NAIVE_CNN_CONFIG_H_
#define NAIVE_CNN_CONFIG_H_

#include <cstddef>

/**
 * define to use exceptions
 */
#define CNN_USE_EXCEPTIONS

#define CNN_TASK_SIZE 8

namespace naive_cnn {

/**
 * calculation data type
 * you can change it to float, or user defined class (fixed point,etc)
 **/
typedef double float_t;

/**
 * size of layer, model, data etc.
 * change to smaller type if memory footprint is severe
 **/
typedef std::size_t cnn_size_t;

}  // namespace naive_cnn

#endif  // NAIVE_CNN_CONFIG_H_

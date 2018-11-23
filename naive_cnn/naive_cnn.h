/*
    Copyright (c) 2018-2019, Lu Feng
    All rights reserved.
*/

#ifndef NAIVE_CNN_NAIVE_CNN_H_
#define NAIVE_CNN_NAIVE_CNN_H_

#include "naive_cnn/config.h"
#include "naive_cnn/network.h"

#include "naive_cnn/layers/convolutional_layer.h"
#include "naive_cnn/layers/fully_connected_layer.h"
#include "naive_cnn/layers/average_pooling_layer.h"
#include "naive_cnn/layers/max_pooling_layer.h"
#include "naive_cnn/layers/linear_layer.h"
#include "naive_cnn/layers/lrn_layer.h"
#include "naive_cnn/layers/dropout_layer.h"
#include "naive_cnn/layers/linear_layer.h"

#include "naive_cnn/activations/activation_function.h"
#include "naive_cnn/lossfunctions/loss_function.h"
#include "naive_cnn/optimizers/optimizer.h"

#include "naive_cnn/util/weight_init.h"
#include "naive_cnn/util/image.h"
#include "naive_cnn/util/deform.h"
#include "naive_cnn/util/product.h"

#include "naive_cnn/io/mnist_parser.h"
#include "naive_cnn/io/cifar10_parser.h"
#include "naive_cnn/io/display.h"
#include "naive_cnn/io/layer_factory.h"

#ifdef CNN_USE_CAFFE_CONVERTER
// experimental / require google protobuf
#include "naive_cnn/io/caffe/layer_factory.h"
#endif

#endif  // NAIVE_CNN_NAIVE_CNN_H_

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(naive_cnn)
set(PROJECT_VERSION "0.0.1")

IF(NOT CMAKE_BUILD_TYPE )
  SET( CMAKE_BUILD_TYPE "Release" )
ENDIF()

OPTION(BUILD_EXAMPLES "Set to ON to build examples" ON)
OPTION(BUILD_TESTS "Set to ON to build tests" OFF)


IF(BUILD_EXAMPLES)
  find_package( OpenCV REQUIRED )
ENDIF()

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR} )
LINK_LIBRARIES(${REQUIRED_LIBRARIES} )

# SET( tiny_cnn_hrds tiny_cnn/activations/activation_function.h
# tiny_cnn/io/cifar10_parser.h  tiny_cnn/layers/convolutional_layer.h
# tiny_cnn/io/display.h  tiny_cnn/util/image.h
# tiny_cnn/layers/layer.h  tiny_cnn/lossfunctions/loss_function.h
# tiny_cnn/io/mnist_parser.h  tiny_cnn/optimizers/optimizer.h  tiny_cnn/util/product.h  tiny_cnn/util/util.h
#   tiny_cnn/layers/average_pooling_layer.h  tiny_cnn/config.h  tiny_cnn/util/deform.h tiny_cnn/layers/fully_connected_layer.h tiny_cnn/layers/input_layer.h  tiny_cnn/layers/layers.h  tiny_cnn/layers/max_pooling_layer.h  tiny_cnn/network.h  tiny_cnn/layers/partial_connected_layer.h  tiny_cnn/tiny_cnn.h  tiny_cnn/util/weight_init.h)

#SET(tiny_cnn_test_headers test/test_average_pooling_layer.h test/test_convolutional_layer.h test/test_fully_connected_layer.h test/test_lrn_layer.h test/test_max_pooling_layer.h test/test_dropout_layer.h test/test_network.h test/test_target_cost.h test/test_large_thread_count.h test/testhelper.h test/picotest/picotest.h)

#IF (BUILD_EXAMPLES)
#  ADD_EXECUTABLE(example_mnist_train examples/mnist/train.cpp ${tiny_cnn_hrds})
#  ADD_EXECUTABLE(example_mnist_test examples/mnist/test.cpp  ${tiny_cnn_hrds})
#  ADD_EXECUTABLE(example_cifar_train examples/cifar10/train.cpp ${tiny_cnn_hrds})
#  target_link_libraries( example_mnist_test ${OpenCV_LIBS} )
#  include_directories(${OpenCV_INCLUDE_DIRS})
#ENDIF()
#
#IF(BUILD_TESTS)
#  ADD_EXECUTABLE(tiny_cnn_test test/test.cpp ${tiny_cnn_hrds} ${tiny_cnn_test_headers})
#ENDIF()

#include "naive_cnn/util/util.h"
#include "naive_cnn/util/image.h"
#include "naive_cnn/layers/partial_connected_layer.h"
#include "naive_cnn/activations/activation_function.h"

namespace naive_cnn {


template<typename Activation = activation::identity>
class average_pooling_layer : public partial_connected_layer<Activation> {
public:
    typedef partial_connected_layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;

    average_pooling_layer(cnn_size_t in_width, cnn_size_t in_height, cnn_size_t in_channels, cnn_size_t pooling_size)
    : Base(in_width * in_height * in_channels, 
           in_width * in_height * in_channels / sqr(pooling_size), 
           in_channels, in_channels, float_t(1) / sqr(pooling_size)),
      stride_(pooling_size),
      in_(in_width, in_height, in_channels), 
      out_(in_width/pooling_size, in_height/pooling_size, in_channels)
    {
        if ((in_width % pooling_size) || (in_height % pooling_size))
            pooling_size_mismatch(in_width, in_height, pooling_size);

        init_connection(pooling_size);
    }

    average_pooling_layer(cnn_size_t in_width, cnn_size_t in_height, cnn_size_t in_channels, cnn_size_t pooling_size, cnn_size_t stride)
        : Base(in_width * in_height * in_channels,
            pool_out_dim(in_width, pooling_size, stride) * pool_out_dim(in_height, pooling_size, stride) * in_channels,
            in_channels, in_channels, float_t(1) / sqr(pooling_size)),
        stride_(stride),
        in_(in_width, in_height, in_channels),
        out_(pool_out_dim(in_width, pooling_size, stride), pool_out_dim(in_height, pooling_size, stride), in_channels)
    {
       // if ((in_width % pooling_size) || (in_height % pooling_size))
       //     pooling_size_mismatch(in_width, in_height, pooling_size);

        init_connection(pooling_size);
    }

    image<> output_to_image(size_t worker_index = 0) const override {
        return vec2image<unsigned char>(Base::get_worker_storage(worker_index).output_, out_);
    }

    index3d<cnn_size_t> in_shape() const override { return in_; }
    index3d<cnn_size_t> out_shape() const override { return out_; }
    std::string layer_type() const override { return "ave-pool"; }

private:
    size_t stride_;

    static cnn_size_t pool_out_dim(cnn_size_t in_size, cnn_size_t pooling_size, cnn_size_t stride) {
        return (int)std::ceil(((double)in_size - pooling_size) / stride) + 1;
    }

    void init_connection(cnn_size_t pooling_size) {
        for (cnn_size_t c = 0; c < in_.depth_; ++c)
            for (cnn_size_t y = 0; y < in_.height_; y += stride_)
                for (cnn_size_t x = 0; x < in_.width_; x += stride_)
                    connect_kernel(pooling_size, x, y, c);


        for (cnn_size_t c = 0; c < in_.depth_; ++c)
            for (cnn_size_t y = 0; y < out_.height_; ++y)
                for (cnn_size_t x = 0; x < out_.width_; ++x)
                    this->connect_bias(c, out_.get_index(x, y, c));
    }

    void connect_kernel(cnn_size_t pooling_size, cnn_size_t x, cnn_size_t y, cnn_size_t inc) {
        cnn_size_t dymax = std::min(pooling_size, in_.height_ - y);
        cnn_size_t dxmax = std::min(pooling_size, in_.width_ - x);
        cnn_size_t dstx = x / stride_;
        cnn_size_t dsty = y / stride_;

        for (cnn_size_t dy = 0; dy < dymax; ++dy)
            for (cnn_size_t dx = 0; dx < dxmax; ++dx)
                this->connect_weight(
                    in_.get_index(x + dx, y + dy, inc), 
                    out_.get_index(dstx, dsty, inc),
                    inc);
    }

    index3d<cnn_size_t> in_;
    index3d<cnn_size_t> out_;
};

} // namespace naive_cnn

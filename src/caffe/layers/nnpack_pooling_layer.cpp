#ifdef USE_NNPACK
#include "nnpack_pooling_layer.hpp"
#include "nnpack.h"

namespace caffe {

template <typename Dtype>
void NNPackPoolingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  return PoolingLayer<Dtype>::Forward_cpu(bottom, top);
}

template <>
void NNPackPoolingLayer<float>::Forward_cpu(
    const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {
  CHECK_EQ(top.size(), 1);
  CHECK_EQ(bottom.size(), 1);

  if (this->layer_param_.pooling_param().pool() !=
    PoolingParameter_PoolMethod_MAX)
  {
    // NNPACK implements only max-pooling
    VLOG(1) << "Falling back to PoolingLayer (non-max pooling)";
    return PoolingLayer<float>::Forward_cpu(bottom, top);
  } else {
    // For max-pooling layers, call NNPACK implementation and if it fails with
    // nnp_unsupported_* status, fall back to the reference implementation
    VLOG(1) << "Using NNPACKPoolingLayer";

    const nnp_size input_size = {
      .width = static_cast<size_t>(bottom[0]->width()),
      .height = static_cast<size_t>(bottom[0]->height()) };
    VLOG(1) << "Input: " << input_size.width << ", " << input_size.height;

    const nnp_padding input_padding = {
        .top = static_cast<size_t>(this->pad_h_),
        .right = static_cast<size_t>(this->pad_w_),
        .bottom = static_cast<size_t>(this->pad_h_),
        .left = static_cast<size_t>(this->pad_w_)};
    VLOG(1) << "Input Padding: " << input_padding.top << ", "
            << input_padding.right;

    const nnp_size pooling_size = {
      .width = static_cast<size_t>(this->kernel_w_),
      .height = static_cast<size_t>(this->kernel_h_) };
    VLOG(1) << "Pooling: " << pooling_size.width << ", " << pooling_size.height;

    const nnp_size pooling_stride = {
      .width = static_cast<size_t>(this->stride_w_),
      .height = static_cast<size_t>(this->stride_h_) };
    VLOG(1) << "Pooling Stride: " << pooling_stride.width << ", "
            << pooling_stride.height;

    const auto status = nnp_max_pooling_output(
        bottom[0]->num(),
        bottom[0]->channels(),
        input_size,
        input_padding,
        pooling_size,
        pooling_stride,
        bottom[0]->cpu_data(),
        top[0]->mutable_cpu_data(),
        Caffe::nnpack_threadpool());
    switch (status) {
      case nnp_status_unsupported_pooling_size:
      case nnp_status_unsupported_pooling_stride:
      case nnp_status_unsupported_hardware:
        VLOG(1) << "Falling back to PoolingLayer (unsupported pooling param)";
        return PoolingLayer<float>::Forward_cpu(bottom, top);
      default:
        CHECK_EQ(status, nnp_status_success);
    }
  }
}


INSTANTIATE_CLASS(NNPackPoolingLayer);

}
#endif

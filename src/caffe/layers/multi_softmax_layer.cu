// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
__global__ void kernel_get_max(const int num, const int dim,
    const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num) {
    Dtype maxval = -FLT_MAX;
    for (int i = 0; i < dim; ++i) {
      maxval = max(data[index * dim + i], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_softmax_div(const int num, const int dim,
    const Dtype* scale, Dtype* data) {
  CUDA_KERNEL_LOOP(index, num * dim) {
    int n = index / dim;
    data[index] /= scale[n];
  }
}

template <typename Dtype>
__global__ void kernel_exp(const int num, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num) {
    out[index] = exp(data[index]);
  }
}

template <typename Dtype>
Dtype MultiSoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
    return Forward_cpu(bottom,top);
}

// TODO(Yangqing): implement the GPU version of softmax.
template <typename Dtype>
void MultiSoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
    
}

INSTANTIATE_CLASS(MultiSoftmaxLayer);


}  // namespace caffe

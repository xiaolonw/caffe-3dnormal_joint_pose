// Copyright 2014 BVLC and contributors.

#include <cublas_v2.h>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype DecodeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Forward_cpu(bottom, top);
  return Dtype(0);
}

template <typename Dtype>
void DecodeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
    Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_CLASS(DecodeLayer);

}  // namespace caffe

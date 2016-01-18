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
Dtype RescaleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
    
    Forward_cpu(bottom, top);
  
  return Dtype(0);
}

// TODO(Yangqing): implement the GPU version of softmax.
template <typename Dtype>
void RescaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
    
    Backward_cpu(top, propagate_down, bottom);
    
}

INSTANTIATE_CLASS(RescaleLayer);

}  // namespace caffe

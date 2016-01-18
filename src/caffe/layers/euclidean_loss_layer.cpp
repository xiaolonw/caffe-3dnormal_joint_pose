// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::FurtherSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  /*CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());*/
  CHECK_EQ(bottom[0]->width() * bottom[0]->height() * bottom[0]->channels(), bottom[1]->width() * bottom[1]->height() *bottom[1]->channels());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
Dtype EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());

  //for 3dnormal invalid point
  /*Dtype* diff_data = diff_.mutable_cpu_data();
  for (int i = 0; i < diff_.count(); i ++)
  {
	  if (fabs(diff_data[i]) > 10)
		  diff_data[i] = 0;
  }*/
  //3dnormal
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  return loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  caffe_cpu_axpby(
      (*bottom)[0]->count(),              // count
      Dtype(1) / (*bottom)[0]->num(),     // alpha
      diff_.cpu_data(),                   // a
      Dtype(0),                           // beta
      (*bottom)[0]->mutable_cpu_diff());  // b
}

INSTANTIATE_CLASS(EuclideanLossLayer);

}  // namespace caffe

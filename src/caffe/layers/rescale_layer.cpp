// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <string>
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

using std::max;

using namespace std;

namespace caffe {

template <typename Dtype>
void RescaleLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "NormLayer Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "NormLayer Layer takes a single blob as output.";
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());

  meanfile.clear();
  const string& source_dict = this->layer_param_.decode_param().source_dict();
  FILE * fid = fopen(source_dict.c_str(), "r");
  float num;
  while(fscanf(fid, "%f", &num) > 0)
  {
	  meanfile.push_back(num);
  }
  fclose(fid);
}


template <typename Dtype>
Dtype RescaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int num = bottom[0]->num();
  const int count = bottom[0]->count();
  int dim = count / num;
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i] * 128 + 128 - meanfile[i % dim];
  }
  return Dtype(0);
}

template <typename Dtype>
void RescaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int count = (*bottom)[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * 128;
    }
  }
}


INSTANTIATE_CLASS(RescaleLayer);


}  // namespace caffe

// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void AngleWithLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "SoftmaxLoss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "SoftmaxLoss Layer takes no blob as output.";
}

template <typename Dtype>
Dtype AngleWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.

  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  int channel = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  int dimScale = height * width;

  Dtype loss = 0;
  for (int i = 0; i < num; ++i)
  {
	  for(int h = 0; h < height; h ++)
	  for(int w = 0; w < width; w ++)
	  {
		  float tnum = 0;
		  // 3dnormal
		  if( fabs(label[i * dim + 0 * dimScale + h * width + w]) > 10 )
		  {
			  continue;
		  }
		  for(int c = 0; c < channel; c ++)
		  {

			  tnum += bottom_data[i * dim + c * dimScale + h * width + w] *
					  label[i * dim + c * dimScale + h * width + w] ;
		  }
		  loss += acos(tnum);
	  }
  }
  return loss / num;
}

template <typename Dtype>
void AngleWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // Compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();

  const Dtype* label = (*bottom)[1]->cpu_data();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  int channel = (*bottom)[0]->channels();
  int height = (*bottom)[0]->height();
  int width = (*bottom)[0]->width();
  int num = (*bottom)[0]->num();
  int dim = (*bottom)[0]->count() / (*bottom)[0]->num();
  int dimScale = height * width;

    for (int i = 0; i < num; ++i)
	{
	  for(int h = 0; h < height; h ++)
	  for(int w = 0; w < width; w ++)
	  {
		  float tnum = 0;
		  // 3dnormal
		  if( fabs(label[i * dim + 0 * dimScale + h * width + w]) > 10 )
		  {
			  for(int c = 0; c < channel; c ++)
			  {
				  bottom_diff[i * dim + c * dimScale + h * width + w] = 0;
			  }
			  continue;
		  }
		  for(int c = 0; c < channel; c ++)
		  {
			  tnum += bottom_data[i * dim + c * dimScale + h * width + w] *
					  label[i * dim + c * dimScale + h * width + w] ;
		  }
		  tnum = 1 / (sqrt(1 - tnum * tnum)  +  1e-6);
		  tnum = - tnum;

		  for(int c = 0; c < channel; c ++)
		  {
			  float ty = label[i * dim + c * dimScale + h * width + w] * tnum ;
			  bottom_diff[i * dim + c * dimScale + h * width + w] = ty;
		  }
	  }
	}

  // Scale down gradient
  caffe_scal((*bottom)[0]->count(), Dtype(1) / num, bottom_diff);
}


INSTANTIATE_CLASS(AngleWithLossLayer);


}  // namespace caffe

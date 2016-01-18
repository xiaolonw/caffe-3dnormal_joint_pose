// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CombineLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  //CHECK_EQ(bottom.size(), 1) << "IP Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "IP Layer takes a single blob as output.";

  int num = bottom[0]->num();
  int out_channel = 0;
  int out_height = bottom[0]->height();
  int out_width = bottom[0]->width();
  for(int i = 0; i < bottom.size(); i ++)
  {
	  out_channel += bottom[i]->channels();
	  int theight = bottom[i]->height();
	  int twidth = bottom[i]->width();
	  CHECK_EQ(out_width,  twidth);
	  CHECK_EQ(out_height, theight);
  }

  (*top)[0]->Reshape(num, out_channel, out_height, out_width);

}

template <typename Dtype>
Dtype CombineLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

    Dtype* top_data = (*top)[0]->mutable_cpu_data();

    int topdim = (*top)[0]->count() / (*top)[0]->num();
    int num = bottom[0]->num();
    int out_channel = 0;
    int out_height = bottom[0]->height();
    int out_width = bottom[0]->width();

    for(int i = 0; i < (*top)[0]->count(); i ++)
	{
		top_data[i] = 0;
	}

    for(int i = 0; i < num; i ++)
    {
    	int cnt = 0;
    	for(int bi = 0; bi < bottom.size(); bi ++)
    	{
    	    const Dtype* bottom_data = bottom[bi]->cpu_data();
    		int dim = bottom[bi]->count() / num;
    		const Dtype* bottom_data_now = bottom_data + dim * i;
        	Dtype* top_data_now = top_data + topdim * i + cnt;
        	for (int k = 0; k < dim; k ++)
        		top_data_now[k] = bottom_data_now[k];

        	cnt += dim;
    	}


    }

    /*for(int i = 0; i < (*top)[0]->count(); i ++)
	{
		top_data[i] -= 110;
	}*/

    return Dtype(0);
}

template <typename Dtype>
void CombineLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {

	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = (*bottom)[0]->cpu_data();

	int topdim = top[0]->count() / top[0]->num();
	int num = top[0]->num();
	int out_channel = 0;
	int out_height = top[0]->height();
	int out_width = top[0]->width();

	for(int i = 0; i < (*bottom).size(); i ++)
	{
		Dtype* bottom_diff = (*bottom)[i]->mutable_cpu_diff();
		for(int j = 0; j < (*bottom)[i]->count(); j ++)
		{
			bottom_diff[j] = 0;
		}
	}

	for(int i = 0; i < num; i ++)
	{
		int cnt = 0;
		for(int bi = 0; bi < (*bottom).size(); bi ++)
		{
			Dtype* bottom_diff = (*bottom)[bi]->mutable_cpu_diff();
			int dim = (*bottom)[bi]->count() / num;
			Dtype* bottom_diff_now = bottom_diff + dim * i;
			const Dtype* top_diff_now = top_diff + topdim * i + cnt;

			for (int k = 0; k < dim; k ++)
				bottom_diff_now[k] = top_diff_now[k];

			cnt += dim;
		}

	}

}

INSTANTIATE_CLASS(CombineLayer);

}  // namespace caffe

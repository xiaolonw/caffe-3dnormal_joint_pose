// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>
#include <iostream>  // NOLINT(readability/streams)
#include <fstream>  // NOLINT(readability/streams)
#include <utility>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

using std::iterator;
using std::string;
using std::pair;

namespace caffe {



bool JointReadImageToDatum(const string& filename, const string &labelfile, const int height, const int width,
		const int height2, const int width2, Datum* datum, Datum* datum2)
{
  cv::Mat cv_img;
  cv::Mat cv_img2;
  if (height > 0 && width > 0)
  {
    cv::Mat cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    cv::resize(cv_img_origin, cv_img,  cv::Size(width, height));
    cv::resize(cv_img_origin, cv_img2, cv::Size(width2, height2));
  }

  if (!cv_img.data || !cv_img2.data)
  {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }

  datum->set_channels(3);
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_label();
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        datum_string->push_back(
            static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
      }
    }
  }

  int num, cnt = 0;
  FILE *fid = fopen(labelfile.c_str(), "r");
  while(fscanf(fid, "%d", &num) > 0)
  {
	  datum->add_label(num);
	  cnt ++;
  }
  fclose(fid);
  CHECK_EQ(cnt, width * height);

  datum2->set_channels(3);
  datum2->set_height(cv_img2.rows);
  datum2->set_width(cv_img2.cols);
  datum2->clear_label();
  datum2->clear_data();
  datum2->clear_float_data();
    string* datum_string2 = datum2->mutable_data();
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < cv_img2.rows; ++h) {
        for (int w = 0; w < cv_img2.cols; ++w) {
        	datum_string2->push_back( static_cast<char>(cv_img2.at<cv::Vec3b>(h, w)[c]));
        }
      }
    }


  return true;
}

template<typename Dtype>
void* JointImageDataLayerPrefetch(void* layer_pointer)
{
	CHECK(layer_pointer);
	JointImageDataLayer<Dtype>* layer = reinterpret_cast<JointImageDataLayer<Dtype>*>(layer_pointer);
	CHECK(layer);
	Datum datum, datum2;
	CHECK(layer->prefetch_data_);
	CHECK(layer->prefetch_data_2_);
	Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
	Dtype* top_data_2 = layer->prefetch_data_2_->mutable_cpu_data();
	Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
	ImageDataParameter image_data_param = layer->layer_param_.image_data_param();
	const Dtype scale = image_data_param.scale();
	const int batch_size = image_data_param.batch_size();
	const int crop_size = image_data_param.crop_size();
	const bool mirror = image_data_param.mirror();
	const int new_height = image_data_param.new_height();
	const int new_width = image_data_param.new_width();
	const int new_height_2 = image_data_param.new_height_2();
	const int new_width_2 = image_data_param.new_width_2();

	if (mirror && crop_size == 0)
	{
		LOG(FATAL)
				<< "Current implementation requires mirror and crop_size to be "
				<< "set at the same time.";
	}
	// datum scales
	const int channels = layer->datum_channels_;
	const int height = layer->datum_height_;
	const int width = layer->datum_width_;
	const int size = layer->datum_size_;
	const int size_2 = layer->datum_size_2_;
	const int lines_size = layer->lines_.size();
	const Dtype* mean = layer->data_mean_.cpu_data();
	const Dtype* mean_2 = layer->data_mean_2_.cpu_data();
	for (int item_id = 0; item_id < batch_size; ++item_id)
	{
		// get a blob
		CHECK_GT(lines_size, layer->lines_id_);
		if (!JointReadImageToDatum(layer->lines_[layer->lines_id_].first, layer->lines_[layer->lines_id_].second,
				new_height, new_width, new_height_2, new_width_2,
				&datum, &datum2))
		{
			continue;
		}
		const string& data = datum.data();
		const string& data2 = datum2.data();
		if (crop_size)
		{
			CHECK_EQ(crop_size, 0 ) << "We do not crop in this version.";

		}
		else
		{
			// Just copy the whole data
			if (data.size())
			{
				for (int j = 0; j < size; ++j)
				{
					Dtype datum_element = static_cast<Dtype>(static_cast<uint8_t>(data[j]));
					top_data[item_id * size + j] = (datum_element - mean[j]) * scale;
				}
				for (int j = 0; j < size_2; ++j)
				{
					Dtype datum_element = static_cast<Dtype>(static_cast<uint8_t>(data2[j]));
					top_data_2[item_id * size_2 + j] = (datum_element - mean_2[j]) * scale;
				}
			}
			else
			{
				for (int j = 0; j < size; ++j)
				{
					top_data[item_id * size + j] = (datum.float_data(j) - mean[j]) * scale;
				}
				for (int j = 0; j < size_2; ++j)
				{
					top_data_2[item_id * size_2 + j] = (datum2.float_data(j) - mean_2[j]) * scale;
				}
			}
		}

		//top_label[item_id] = datum.label();
		for (int label_i = 0; label_i < datum.label_size(); label_i++)
		{
			top_label[item_id * datum.label_size() + label_i] = datum.label(label_i);
		}
		// go to the next iter
		layer->lines_id_++;
		if (layer->lines_id_ >= lines_size)
		{
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			layer->lines_id_ = 0;
			if (layer->layer_param_.image_data_param().shuffle())
			{
				layer->ShuffleImages();
			}
		}
	}

	return reinterpret_cast<void*>(NULL);
}

template <typename Dtype>
JointImageDataLayer<Dtype>::~JointImageDataLayer<Dtype>() {
  JoinPrefetchThread();
}

template <typename Dtype>
void JointImageDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "Input Layer takes no input blobs.";
  CHECK_EQ(top->size(), 3) << "Input Layer takes two blobs as output.";
  const int new_height  = this->layer_param_.image_data_param().new_height();
  const int new_width  =  this->layer_param_.image_data_param().new_width();
  const int new_height_2  = this->layer_param_.image_data_param().new_height_2();
  const int new_width_2  =  this->layer_param_.image_data_param().new_width_2();
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename, labelname;
  while (infile >> filename >> labelname) {
    lines_.push_back(std::make_pair(filename, labelname));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Never shuffle here.";
    //const unsigned int prefetch_rng_seed = caffe_rng_rand();
    //prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    //ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  CHECK(ReadImageToDatum(lines_[lines_id_].first, 0, new_height, new_width, &datum));
  // image
  const int crop_size = this->layer_param_.image_data_param().crop_size();
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  const string& mean_file = this->layer_param_.image_data_param().mean_file();
  const string& mean_file_2 = this->layer_param_.image_data_param().mean_file_2();
  if (crop_size > 0)
  {
    (*top)[0]->Reshape(batch_size, datum.channels(), new_height, new_width);
    (*top)[2]->Reshape(batch_size, datum.channels(), new_height_2, new_width_2);
    prefetch_data_.reset(new Blob<Dtype>(batch_size, datum.channels(), new_height, new_width));
    prefetch_data_2_.reset(new Blob<Dtype>(batch_size, datum.channels(), new_height_2, new_width_2));
  }
  else
  {
    (*top)[0]->Reshape(batch_size, datum.channels(), datum.height(), datum.width());
    (*top)[2]->Reshape(batch_size, datum.channels(), new_height_2, new_width_2);
    prefetch_data_.reset(new Blob<Dtype>(batch_size, datum.channels(), datum.height(), datum.width()));
    prefetch_data_2_.reset(new Blob<Dtype>(batch_size, datum.channels(), new_height_2, new_width_2));
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  (*top)[1]->Reshape(batch_size, 1, new_height, new_width);
  prefetch_label_.reset(new Blob<Dtype>(batch_size, 1, new_height, new_width));
  // datum size
  datum_channels_ = datum.channels();
  datum_height_ = datum.height();
  datum_width_ = datum.width();
  datum_size_ = datum.channels() * datum.height() * datum.width();

    datum_height_2_ = new_height_2;
    datum_width_2_ = new_width_2;
    datum_size_2_ = datum.channels() * new_height_2 * new_width_2;

  //CHECK_GT(datum_height_, crop_size);
  //CHECK_GT(datum_width_, crop_size);
  // check if we want to have mean
  if (this->layer_param_.image_data_param().has_mean_file()) {
    BlobProto blob_proto;
    LOG(INFO) << "Loading mean file from" << mean_file;
    ReadProtoFromBinaryFile(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);

    BlobProto blob_proto_2;
	LOG(INFO) << "Loading mean file from" << mean_file_2;
	ReadProtoFromBinaryFile(mean_file_2.c_str(), &blob_proto_2);
	data_mean_2_.FromProto(blob_proto_2);

    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.channels(), datum_channels_);
    CHECK_EQ(data_mean_.height(), datum_height_);
    CHECK_EQ(data_mean_.width(), datum_width_);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  prefetch_data_2_->mutable_cpu_data();
  prefetch_label_->mutable_cpu_data();
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void JointImageDataLayer<Dtype>::CreatePrefetchThread() {
  phase_ = Caffe::phase();
  const bool prefetch_needs_rand =
      this->layer_param_.image_data_param().shuffle() ||
          ((phase_ == Caffe::TRAIN) &&
           (this->layer_param_.image_data_param().mirror() ||
            this->layer_param_.image_data_param().crop_size()));
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }
  // Create the thread.
  CHECK(!pthread_create(&thread_, NULL, JointImageDataLayerPrefetch<Dtype>,
        static_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void JointImageDataLayer<Dtype>::ShuffleImages() {
  const int num_images = lines_.size();
  LOG(INFO) << "Never shuffle here.";
  /*for (int i = 0; i < num_images; ++i) {
    const int max_rand_index = num_images - i;
    const int rand_index = PrefetchRand() % max_rand_index;
    pair<string, int> item = lines_[rand_index];
    lines_.erase(lines_.begin() + rand_index);
    lines_.push_back(item);
  }*/
}

template <typename Dtype>
void JointImageDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
unsigned int JointImageDataLayer<Dtype>::PrefetchRand() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
Dtype JointImageDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_->count(), prefetch_data_->cpu_data(),
             (*top)[0]->mutable_cpu_data());
  caffe_copy(prefetch_data_2_->count(), prefetch_data_2_->cpu_data(),
               (*top)[2]->mutable_cpu_data());
  caffe_copy(prefetch_label_->count(), prefetch_label_->cpu_data(),
             (*top)[1]->mutable_cpu_data());
  // Start a new prefetch thread
  CreatePrefetchThread();
  return Dtype(0.);
}

INSTANTIATE_CLASS(JointImageDataLayer);

}  // namespace caffe

// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>
#include <iostream>  // NOLINT(readability/streams)
#include <fstream>  // NOLINT(readability/streams)
#include <utility>

#include <cv.h>
#include <highgui.h>
#include <cxcore.h>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

using std::iterator;
using std::string;
using std::pair;

using namespace cv;
using namespace std;

namespace caffe {

float random(float start, float end)
{
        return start+(end-start)*caffe_rng_rand()/(UINT_MAX + 1.0);
}

template<typename Dtype>
bool PoseReadLabel(const vector<int> & lbls, int * was, Dtype * lblmap, const int out_height, const int out_width)
{
	int pointNum = lbls.size() / 2 - 2;
	for(int i = 0; i < out_height * out_width; i ++) was[i] = 0;
	for(int i = 0; i < out_height * out_width * pointNum ; i ++) lblmap[i] = 0;

	vector<int> dx;
	vector<int> dy;
	dx.push_back(-2); dy.push_back(0);
	dx.push_back(2); dy.push_back(0);
	dy.push_back(-2); dx.push_back(0);
	dy.push_back(2); dx.push_back(0);
	for(int t1 = -1; t1 <= 1; t1 ++)
		for(int t2 = -1; t2 <= 1; t2 ++)
		{
			dx.push_back(t1);
			dy.push_back(t2);
		}

	vector<int> dx2;
	vector<int> dy2;
	for(int t1 = -2; t1 <= 2; t1 ++)
		for(int t2 = -2; t2 <= 2; t2 ++)
		{
			dx2.push_back(t1);
			dy2.push_back(t2);
		}
	for(int i = 0; i < pointNum * 2; i += 2)
	{
		int nowx = lbls[i];
		int nowy = lbls[i + 1];
		int chn  = i / 2;
		for(int j = 0; j < dx.size(); j ++)
		{
			int tx = nowx + dx[j], ty = nowy + dy[j];
			if(tx >= out_width || tx < 0 || ty >= out_height || ty < 0) continue;
			was[ty * out_width + tx] = 1 + chn;
			lblmap[chn * out_height * out_width + ty * out_width + tx] = 1;
		}
	}
	for(int h = 0; h < out_height; h ++)
		for(int w = 0; w < out_width; w ++)
		{
			for(int c = 0; c < pointNum; c ++)
			{
				if(lblmap[c * out_height * out_width + h * out_width + w] > 0)
					continue;
				int flag = 0;
				for(int j = 0; j < dx2.size(); j ++)
				{
					int tx = w + dx2[j], ty = h + dy2[j];
					if(tx >= out_width || tx < 0 || ty >= out_height || ty < 0) continue;
					if(lblmap[c * out_height * out_width + ty * out_width + tx]  > 0 )
					{
						flag = 1;
						break;
					}
				}
				if(flag)
				{
					lblmap[c * out_height * out_width + h * out_width + w] = -1;
				}
			}
		}

	return true;

}


bool PoseReadImageToDatum(const string& filename, const string &labelfile, const int height, const int width, Datum* datum, const float scale,
		const float torso_ratio, const int mx1, const int mx2, const int my1, const int my2)
{
  cv::Mat cv_img;
  if (height > 0 && width > 0)
  {
    cv_img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    //cv::resize(cv_img_origin, cv_img,  cv::Size(width, height));
  }

  if (!cv_img.data )
  {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }

  datum->set_channels(3);
  datum->set_height(height);
  datum->set_width(width);
  datum->clear_label();
  datum->clear_data();
  datum->clear_float_data();

    vector<int> pts;
    int num;
	FILE *fid = fopen(labelfile.c_str(), "r");
	while(fscanf(fid, "%d", &num) > 0)
	{
		pts.push_back(num);
	  //datum->add_label(num);
	}
	fclose(fid);
	int pointNum = pts.size() / 2 - 2;


	// normalize according to torso size: height * 0.7 / torsolen
	float tb4 = pts[pts.size() - 1], tb3 = pts[pts.size() - 2], tb2 = pts[pts.size() - 3], tb1 = pts[pts.size() - 4];
	float torsolen = sqrt((tb4 - tb2)*(tb4 - tb2) + (tb3 - tb1)*(tb3 - tb1));
	float tratio = height * torso_ratio / torsolen;

	for(int i = 0; i < pts.size(); i ++)
	{
		pts[i] = pts[i] * tratio;
	}
	int newh = cv_img.rows * tratio;
	int neww = cv_img.cols * tratio;
	cv::Mat cv_img2(Size(neww, newh),CV_8UC3);
	cv::resize(cv_img, cv_img2, cv::Size(neww, newh));

	int minx = 1e6, miny = 1e6;
	int maxx = 0, maxy = 0;

	for (int label_i = 0; label_i < pointNum * 2; label_i += 2)
	{
		int tx = pts[label_i];
		int ty = pts[label_i + 1];
		minx = min(minx, tx);
		miny = min(miny, ty);
		maxx = max(maxx, tx);
		maxy = max(maxy, ty);
	}

	// scale
	int height2 = (maxy - miny) * scale;
	int width2 = (maxx - minx) * scale;
	int centx = (minx + maxx) / 2;
	int centy = (miny + maxy) / 2;

	// add margin
	int x1 = max(centx - width2  / 2 - mx1, 0);
	int y1 = max(centy - height2 / 2 - my1, 0);
	int x2 = min(centx + width2  / 2 + mx2, cv_img2.cols);
	int y2 = min(centy + height2 / 2 + my2, cv_img2.rows);

	height2 = y2 - y1;
	width2  = x2 - x1;

	for (int label_i = 0; label_i < pts.size(); label_i += 2)
	{
		pts[label_i] = pts[label_i] - x1;
		pts[label_i + 1] = pts[label_i + 1] - y1;
	}

  Mat imgnow(Size(width2, height2), CV_8UC3);

  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < height2; ++h) {
      for (int w = 0; w < width2; ++w) {
    	  imgnow.at<cv::Vec3b>(h, w)[c]= cv_img2.at<cv::Vec3b>(h + y1, w + x1)[c];
      }
    }
  }

  float hratio = (height * 1.0) / (height2 * 1.0);
  float wratio = (width * 1.0) / (width2 * 1.0);
  float nowratio = min(hratio, wratio);

  if(nowratio < 1)
  {
	  height2 = height2 * nowratio;
	  width2  = width2 * nowratio;
	  Mat imgnow2(Size(width2, height2),CV_8UC3);
	  cv::resize(imgnow, imgnow2, cv::Size(width2, height2) );
	  imgnow = imgnow2;
	  for (int label_i = 0; label_i < pts.size(); label_i += 2)
	  {
		  pts[label_i] = pts[label_i] * nowratio;
		  pts[label_i + 1] = pts[label_i + 1] * nowratio;
	  }
  }

  Mat imgresize(Size(width, height), CV_8UC3);
  int offw = max((width - width2) / 2 - 1, 0);
  int offh = 0;

  for (int label_i = 0; label_i < pts.size(); label_i += 2)
  {
	  pts[label_i] = pts[label_i]  + offw;
	  pts[label_i + 1] = pts[label_i + 1]  + offh;
  }

  for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            imgresize.at<cv::Vec3b>(h, w)[c] = 0;
          }
        }
      }

  for (int c = 0; c < 3; ++c) {
        for (int h = offh; h < height2 + offh; ++h) {
          for (int w = offw; w < width2 + offw; ++w) {
          	imgresize.at<cv::Vec3b>(h, w)[c] = imgnow.at<cv::Vec3b>(h - offh, w - offw)[c];
          }
        }
      }

  string* datum_string = datum->mutable_data();
  for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
        	datum_string->push_back(
        	            static_cast<char>(imgresize.at<cv::Vec3b>(h, w)[c]));
        }
      }
    }

  for (int label_i = 0; label_i < pts.size(); label_i += 2)
  {
		datum->add_label(pts[label_i]);
		datum->add_label(pts[label_i + 1]);
  }


  //CHECK_EQ(cnt, width * height);


  return true;
}

template<typename Dtype>
void* PoseImageDataLayerPrefetch(void* layer_pointer)
{
	CHECK(layer_pointer);
	PoseImageDataLayer<Dtype>* layer =
			reinterpret_cast<PoseImageDataLayer<Dtype>*>(layer_pointer);
	CHECK(layer);
	Datum datum;
	CHECK(layer->prefetch_data_);
	Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
	Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
	PoseImageDataParameter pose_image_data_param = layer->layer_param_.pose_image_data_param();
	const Dtype scale = pose_image_data_param.scale();
	const int batch_size = pose_image_data_param.batch_size();
	const int crop_size = pose_image_data_param.crop_size();
	const bool mirror = pose_image_data_param.mirror();
	const int new_height = pose_image_data_param.new_height();
	const int new_width = pose_image_data_param.new_width();
	const int out_height = pose_image_data_param.out_height();
	const int out_width  = pose_image_data_param.out_width();
	const int key_point_range = pose_image_data_param.key_point_range();
	const float scale_lower_bound = pose_image_data_param.scale_lower_bound();
	const float scale_upper_bound = pose_image_data_param.scale_upper_bound();
	const int key_point_num  = pose_image_data_param.key_point_num();

	const float torso_ratio = pose_image_data_param.torso_ratio();
	const int mx1 = pose_image_data_param.mx1();
	const int mx2 = pose_image_data_param.mx2();
	const int my1 = pose_image_data_param.my1();
	const int my2 = pose_image_data_param.my2();


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
	const int lines_size = layer->lines_.size();
	const Dtype* mean = layer->data_mean_.cpu_data();

	int * was = new int[out_height * out_width];

	for (int item_id = 0; item_id < batch_size; ++item_id)
	{
		//char ss1[1010],ss2[1010];
		//sprintf(ss1,"/home/dragon123/cnncode/showimg/%d.jpg",item_id);
		//sprintf(ss2,"/home/dragon123/cnncode/showimg/%d_gt.jpg",item_id);
		// get a blob
		float nowscale = 1;
		if (layer->phase_ == Caffe::TRAIN)
			nowscale = random(scale_lower_bound, scale_upper_bound);
		CHECK_GT(1.55, nowscale);
		CHECK_GT(nowscale, 0.95);

		CHECK_GT(lines_size, layer->lines_id_);
		if (!PoseReadImageToDatum(layer->lines_[layer->lines_id_].first,
				layer->lines_[layer->lines_id_].second, new_height, new_width, &datum, nowscale,
				torso_ratio, mx1, mx2, my1, my2))
		{
			continue;
		}
		const string& data = datum.data();

		if (new_height > 0 && new_width > 0)
		{
			CHECK(data.size()) << "Image cropping only support uint8 data";
			int h_off, w_off;
			// We only do random crop when we do training.
			h_off = 0;
			w_off = 0;

			if (mirror && layer->PrefetchRand() % 2)
			{
				// Copy mirrored version
				for (int c = 0; c < channels; ++c)
				{
					for (int h = 0; h < new_height; ++h)
					{
						for (int w = 0; w < new_width; ++w)
						{
							int top_index = ((item_id * channels + c)
									* new_height + h) * new_width
									+ (new_width - 1 - w);
							int data_index = (c * height + h + h_off) * width
									+ w + w_off;
							Dtype datum_element =
									static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
							top_data[top_index] = (datum_element
									- mean[data_index]) * scale;
						}
					}
				}
			}
			else
			{
				// Normal copy
				//Mat img(Size(240,240), CV_8UC3);
				for (int c = 0; c < channels; ++c)
				{
					for (int h = 0; h < new_height; ++h)
					{
						for (int w = 0; w < new_width; ++w)
						{
							int top_index = ((item_id * channels + c)
									* new_height + h) * new_width + w;
							int data_index = (c * height + h + h_off) * width
									+ w + w_off;
							Dtype datum_element =
									static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
							top_data[top_index] = (datum_element
									- mean[data_index]) * scale;

							//img.at<cv::Vec3b>(h, w)[c] = (uchar)(datum_element * scale);
						}
					}
				}
				//imwrite(ss1, img);
			}
		}
		else
		{
			// Just copy the whole data
			if (data.size())
			{
				for (int j = 0; j < size; ++j)
				{
					Dtype datum_element =
							static_cast<Dtype>(static_cast<uint8_t>(data[j]));
					top_data[item_id * size + j] = (datum_element - mean[j])
							* scale;
				}
			}
			else
			{
				for (int j = 0; j < size; ++j)
				{
					top_data[item_id * size + j] = (datum.float_data(j)
							- mean[j]) * scale;
				}
			}
		}

		float lblratio = new_height / out_height;
		vector<int> pts;
		for (int label_i = 0; label_i < datum.label_size(); label_i++)
		{
			pts.push_back( datum.label(label_i) / lblratio );
		}

		int lblLen = key_point_num * out_height * out_width;
		PoseReadLabel(pts, was, top_label + item_id * lblLen, out_height, out_width);

		/*for(int ci = 0; ci < key_point_num; ci ++)
		{
			Mat img(Size(out_height, out_width), CV_8UC3);
			sprintf(ss2,"/home/dragon123/cnncode/showimg/%d_%d_gt.jpg",item_id, ci);
			for(int h = 0; h < out_height; h ++)
				for(int w = 0; w < out_width; w ++)
				{
					int clr = top_label[item_id * lblLen + ci * out_height * out_width + h * out_width + w];
					if(clr <= 0)
					{
						if(clr == 0) for(int c = 0; c < 3; c ++) img.at<cv::Vec3b>(h, w)[c] = 0;
						if(clr < 0) for(int c = 0; c < 3; c ++) img.at<cv::Vec3b>(h, w)[c] = 128;
					}
					else
					{
						for(int c = 0; c < 3; c ++) img.at<cv::Vec3b>(h, w)[c] = 255;
					}
 				}
			imwrite(ss2, img);
		}*/


		// go to the next iter
		layer->lines_id_++;
		if (layer->lines_id_ >= lines_size)
		{
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			layer->lines_id_ = 0;
			if (layer->layer_param_.pose_image_data_param().shuffle())
			{
				layer->ShuffleImages();
			}
		}
	}

	delete was;

	return reinterpret_cast<void*>(NULL);
}

template <typename Dtype>
PoseImageDataLayer<Dtype>::~PoseImageDataLayer<Dtype>() {
  JoinPrefetchThread();
}

template <typename Dtype>
void PoseImageDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  //srand(unsigned(time(0)));
  CHECK_EQ(bottom.size(), 0) << "Input Layer takes no input blobs.";
  CHECK_EQ(top->size(), 2) << "Input Layer takes two blobs as output.";
  const int new_height  = this->layer_param_.pose_image_data_param().new_height();
  const int new_width  = this->layer_param_.pose_image_data_param().new_width();
  const int out_height  = this->layer_param_.pose_image_data_param().out_height();
  const int out_width  = this->layer_param_.pose_image_data_param().out_width();
  const int key_point_num  = this->layer_param_.pose_image_data_param().key_point_num();
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.pose_image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename, labelname;
  while (infile >> filename >> labelname) {
      lines_.push_back(std::make_pair(filename, labelname));
  }

  if (this->layer_param_.pose_image_data_param().shuffle()) {
    // randomly shuffle data
	  LOG(INFO) << "Shuffling data";
	  const unsigned int prefetch_rng_seed = caffe_rng_rand();
	  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
	  ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.pose_image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.pose_image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  CHECK(ReadImageToDatum(lines_[lines_id_].first, 0, new_height, new_width, &datum));
  // image
  const int crop_size = this->layer_param_.pose_image_data_param().crop_size();
  const int batch_size = this->layer_param_.pose_image_data_param().batch_size();
  const string& mean_file = this->layer_param_.pose_image_data_param().mean_file();
  if (crop_size > 0) {
	(*top)[0]->Reshape(batch_size, datum.channels(), new_height, new_width);
    prefetch_data_.reset(new Blob<Dtype>(batch_size, datum.channels(), new_height, new_width));
  } else {
	(*top)[0]->Reshape(batch_size, datum.channels(), new_height, new_width);
	prefetch_data_.reset(new Blob<Dtype>(batch_size, datum.channels(), new_height, new_width));
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label

  (*top)[1]->Reshape(batch_size, key_point_num, out_height, out_width);
  prefetch_label_.reset(new Blob<Dtype>(batch_size, key_point_num, out_height, out_width));

  // datum size
  datum_channels_ = datum.channels();
  datum_height_ = new_height;
  datum_width_ = new_width;
  datum_size_ = datum.channels() * new_height * new_width;
  //CHECK_GT(datum_height_, crop_size);
  //CHECK_GT(datum_width_, crop_size);
  // check if we want to have mean
  if (this->layer_param_.pose_image_data_param().has_mean_file()) {
    BlobProto blob_proto;
    LOG(INFO) << "Loading mean file from" << mean_file;
    ReadProtoFromBinaryFile(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.channels(), datum_channels_);
    CHECK_EQ(data_mean_.height(), datum_height_);
    CHECK_EQ(data_mean_.width(), datum_width_);

    /*FILE * file = fopen("/home/dragon123/pose_cnncode/videomean.txt", "w");
	for(int c = 0; c < datum_channels_; c ++)
	for(int h = 0; h < datum_height_; h ++)
		for(int w =0 ; w < datum_width_; w ++)
		{
			fprintf(file, "%f\n", data_mean_.data_at(0,c,h,w));
		}
	fclose(file);*/


  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  prefetch_label_->mutable_cpu_data();
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void PoseImageDataLayer<Dtype>::CreatePrefetchThread() {
  phase_ = Caffe::phase();
  const bool prefetch_needs_rand =
      this->layer_param_.pose_image_data_param().shuffle() ||
          ((phase_ == Caffe::TRAIN) &&
           (this->layer_param_.pose_image_data_param().mirror() ||
            this->layer_param_.pose_image_data_param().crop_size()));
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }
  // Create the thread.
  CHECK(!pthread_create(&thread_, NULL, PoseImageDataLayerPrefetch<Dtype>,
        static_cast<void*>(this))) << "Pthread execution failed.";
}


int myrandomdata (int i) { return caffe_rng_rand()%i;}

template <typename Dtype>
void PoseImageDataLayer<Dtype>::ShuffleImages() {
  const int num_images = lines_.size();
  DLOG(INFO) << "My Shuffle.";
  vector<std::pair<std::string, std::string> > tlines_;
  vector<int> tnum;

  for(int i = 0; i < num_images; i ++)
  {
	  tnum.push_back(i);
  }
  std::random_shuffle(tnum.begin(), tnum.end(), myrandomdata);
  tlines_.clear();
  for(int i = 0; i < num_images; i ++)
  {
	  tlines_.push_back(lines_[tnum[i]]);
  }
  lines_ = tlines_;

  /*for (int i = 0; i < num_images; ++i) {
    const int max_rand_index = num_images - i;
    const int rand_index = PrefetchRand() % max_rand_index;
    pair<string, int> item = lines_[rand_index];
    lines_.erase(lines_.begin() + rand_index);
    lines_.push_back(item);
  }*/
}

template <typename Dtype>
void PoseImageDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
unsigned int PoseImageDataLayer<Dtype>::PrefetchRand() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
Dtype PoseImageDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_->count(), prefetch_data_->cpu_data(),
             (*top)[0]->mutable_cpu_data());
  caffe_copy(prefetch_label_->count(), prefetch_label_->cpu_data(),
             (*top)[1]->mutable_cpu_data());
  // Start a new prefetch thread
  CreatePrefetchThread();
  return Dtype(0.);
}

INSTANTIATE_CLASS(PoseImageDataLayer);

}  // namespace caffe

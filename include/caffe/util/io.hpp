// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

//#include <string>

#include <stdint.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <string>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)

#include "google/protobuf/message.h"
#include "hdf5.h"
#include "hdf5_hl.h"
#include "caffe/proto/caffe.pb.h"

#include "caffe/blob.hpp"

using std::string;
using ::google::protobuf::Message;

#define HDF5_NUM_DIMS 4

namespace caffe {

bool ReadProtoFromTextFile(const char* filename, Message* proto);

inline bool ReadProtoFromTextFile(const string& filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto));
}

inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

inline void ReadProtoFromBinaryFileOrDie(const string& filename,
                                         Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}


void WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}


template <class T>
inline bool ReadImageToDatum(const string& filename, const T label,
    const int height, const int width, Datum* datum) {
  cv::Mat cv_img;
  if (height > 0 && width > 0) {
    cv::Mat cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  }
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }
  datum->set_channels(3);
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  //datum->set_label(label);
  datum->clear_label();
  datum->add_label(label);
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
  return true;
}

template <class T>
inline bool ReadImageToDatum(const string& filename, const T label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, datum);
}

template <typename Dtype>
void hdf5_load_nd_dataset_helper(
  hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
  Blob<Dtype>* blob);

template <typename Dtype>
void hdf5_load_nd_dataset(
  hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
  Blob<Dtype>* blob);

template <typename Dtype>
void hdf5_save_nd_dataset(
  const hid_t file_id, const string dataset_name, const Blob<Dtype>& blob);

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_

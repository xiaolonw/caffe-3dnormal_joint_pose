/*
 * convert_normal.cpp
 *
 *  Created on: Aug 11, 2014
 *      Author: dragon123
 */

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>

#include <cv.h>
#include <highgui.h>
#include <cxcore.h>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;
using std::string;

using namespace cv;


#define RESIZE_HEIGHT 48
#define RESIZE_WIDTH 64
#define LABEL_LEN 9216

// use float label, if changed, one should change caffe.proto label too

struct Seg_Anno {
	string filename_;
	std::vector<float> pos_;
};

bool MyReadImageToDatum(const string& filename, const string &filename2, const std::vector<float> & label,
    const int height, const int width, Datum* datum)
{
	cv::Mat cv_img;
	if (height > 0 && width > 0)
	{
		cv::Mat cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
		cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
	}
	else
	{
		cv_img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	}

	//imshow("pic",cv_img);
	//waitKey(30);

	if (!cv_img.data)
	{
		LOG(ERROR) << "Could not open or find file " << filename;
		return false;
	}
	datum->set_channels(6);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	datum->clear_data();
	datum->clear_float_data();

	for (int c = 0; c < 3; ++c) {
		for (int h = 0; h < cv_img.rows; ++h) {
		  for (int w = 0; w < cv_img.cols; ++w) {
			datum->add_float_data(static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
		  }
		}
	}

	//show for debug
	//char ss2[1010];
	//sprintf(ss2,"/home/dragon123/cnncode/3ddata/testReg/%c%c_coarse.jpg",filename[filename.size() - 10],filename[filename.size() - 9]);
	//Mat imgIn(Size(RESIZE_WIDTH,RESIZE_HEIGHT),CV_8UC3);

	FILE *pFile = fopen(filename2.c_str(), "rb");
	for (int c = 0; c < 3; ++c)
	{
		for (int h = 0; h < cv_img.rows; ++h)
		{
		  for (int w = 0; w < cv_img.cols; ++w)
		  {
			float tnum;
			fread(&tnum, sizeof(float), 1, pFile);
			datum->add_float_data(tnum);
	//		imgIn.at<cv::Vec3b>(h, w)[2 - c] = (uchar)(tnum);
		  }
		}
	}
	fclose(pFile);

	//imwrite(ss2,imgIn);



	//char ss1[1010];
	//sprintf(ss1,"/home/dragon123/cnncode/3ddata/testReg/%c%c_label.jpg",filename[filename.size() - 10],filename[filename.size() - 9]);
	//Mat imgOut(Size(RESIZE_WIDTH,RESIZE_HEIGHT),CV_8UC3);

	//datum->set_label(label);
	datum->clear_label();
	for(int i = 0; i < label.size(); i ++)
	{
		int cnum = i / (RESIZE_HEIGHT * RESIZE_WIDTH);
		int hnum = (i - cnum * RESIZE_HEIGHT * RESIZE_WIDTH) / RESIZE_WIDTH;
		int wnum = i - cnum * RESIZE_HEIGHT * RESIZE_WIDTH - hnum * RESIZE_WIDTH;
		int matNum = cnum * RESIZE_HEIGHT * RESIZE_WIDTH + wnum * RESIZE_HEIGHT + hnum;
		datum->add_label(label[matNum]);
	//	imgOut.at<cv::Vec3b>(hnum, wnum)[2 - cnum] = (uchar)(label[matNum] * 128 + 128);
	}
	//imwrite(ss1,imgOut);

	return true;
}


int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	if (argc < 5) {
		printf(
				"Convert a set of images to the leveldb format used\n"
						"as input for Caffe.\n"
						"Usage:\n"
						"    convert_imageset ROOTFOLDER/ ANNOTATION DB_NAME"
						" MODE[0-train, 1-val, 2-test] RANDOM_SHUFFLE_DATA[0 or 1, default 1] RESIZE_WIDTH[default 256] RESIZE_HEIGHT[default 256](0 indicates no resize)\n"
						"The ImageNet dataset for the training demo is at\n"
						"    http://www.image-net.org/download-images\n");
		return 0;
	}
	std::ifstream infile(argv[2]);
	string root_folder(argv[1]);
	string coarse_folder(argv[8]);
	std::vector<Seg_Anno> annos;
	std::set<string> fNames;
	string filename;
	float prop;
	while (infile >> filename)
	{
		LOG(INFO)<<filename;

		Seg_Anno seg_Anno;
		seg_Anno.filename_ = filename;
		for (int i = 0; i < LABEL_LEN; i++)
		{
			infile >> prop;
			if(!(prop < 1000000 && prop > -1000000))
			{
				printf("123");
			}
			seg_Anno.pos_.push_back(prop);
		}
		string labelFile = filename;
		labelFile[labelFile.size() - 1] = 't';
		labelFile[labelFile.size() - 2] = 'x';
		labelFile[labelFile.size() - 3] = 't';
		labelFile =  coarse_folder + "/" + labelFile;
		//std::ifstream fin(labelFile.c_str());
		FILE * tf = fopen(labelFile.c_str(), "rb");
		if(tf == NULL) continue;
		fclose(tf);
		if (fNames.find(filename)== fNames.end())
		{
			fNames.insert(filename);
			annos.push_back(seg_Anno);
		}
		//debug
		//if(annos.size() == 10)
		//	break;
	}
	if (argc < 6 || argv[5][0] != '0') {
		// randomly shuffle data
		LOG(INFO)<< "Shuffling data";
		std::random_shuffle(annos.begin(), annos.end());
	}
	LOG(INFO)<< "A total of " << annos.size() << " images.";

	leveldb::DB* db;
	leveldb::Options options;
	options.error_if_exists = true;
	options.create_if_missing = true;
	options.write_buffer_size = 268435456;
	LOG(INFO)<< "Opening leveldb " << argv[3];
	leveldb::Status status = leveldb::DB::Open(options, argv[3], &db);
	CHECK(status.ok()) << "Failed to open leveldb " << argv[3];

	Datum datum;
	int count = 0;
	const int maxKeyLength = 256;
	char key_cstr[maxKeyLength];
	leveldb::WriteBatch* batch = new leveldb::WriteBatch();
	int data_size;
	bool data_size_initialized = false;

	// resize to height * width
    int width = RESIZE_WIDTH;
    int height = RESIZE_HEIGHT;
    if (argc > 6) width = atoi(argv[6]);
    if (argc > 7) height = atoi(argv[7]);
    if (width == 0 || height == 0)
        LOG(INFO) << "NO RESIZE SHOULD BE DONE";
    else
        LOG(INFO) << "RESIZE DIM: " << width << "*" << height;

	for (int anno_id = 0; anno_id < annos.size(); ++anno_id)
	{
		string labelFile = annos[anno_id].filename_;
		labelFile[labelFile.size() - 1] = 't';
		labelFile[labelFile.size() - 2] = 'x';
		labelFile[labelFile.size() - 3] = 't';
		if (!MyReadImageToDatum(root_folder + "/" + annos[anno_id].filename_, coarse_folder + "/" + labelFile,
				annos[anno_id].pos_, height, width, &datum))
		{
			continue;
		}
		if (!data_size_initialized)
		{
			data_size = datum.channels() * datum.height() * datum.width() ;
			data_size_initialized = true;
		}
		else
		{
			int dataLen = datum.float_data_size();
			CHECK_EQ(dataLen, data_size)<< "Incorrect data field size " << dataLen;
		}

		// sequential
		snprintf(key_cstr, maxKeyLength, "%07d_%s", anno_id, annos[anno_id].filename_.c_str());
		string value;
		// get the value
		datum.SerializeToString(&value);
		batch->Put(string(key_cstr), value);
		if (++count % 1000 == 0)
		{
			db->Write(leveldb::WriteOptions(), batch);
			LOG(ERROR)<< "Processed " << count << " files.";
			delete batch;
			batch = new leveldb::WriteBatch();
		}
	}
	// write the last batch
	if (count % 1000 != 0) {
		db->Write(leveldb::WriteOptions(), batch);
		LOG(ERROR)<< "Processed " << count << " files.";
	}

	delete batch;
	delete db;
	return 0;
}

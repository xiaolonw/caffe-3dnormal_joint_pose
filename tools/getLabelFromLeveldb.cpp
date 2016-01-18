/*
 * getLabelFromLeveldb.cpp
 *
 *  Created on: Aug 18, 2014
 *      Author: dragon123
 */


#include <cv.h>
#include <highgui.h>
#include <cxcore.h>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <leveldb/db.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"

#include <string>
#include <fstream>

using namespace std;
using namespace cv;
using namespace caffe;
using std::vector;
#define LABEL_LEN 3
#define LABEL_HEIGHT 55
#define LABEL_WIDTH 55
#define IMG_LENGTH 256




int readLeveldb(string &source, string &des)
{
	int counts_ = 0;
	leveldb::DB* db_temp;
	leveldb::Options options;
	options.create_if_missing = false;
	options.max_open_files = 100;
	LOG(INFO) << "Opening leveldb " << source;
	leveldb::Status status = leveldb::DB::Open(options, source, &db_temp);
	CHECK(status.ok()) << "Failed to open leveldb ";

	shared_ptr<leveldb::DB> db_;
	shared_ptr<leveldb::Iterator> iter_;
	db_.reset(db_temp);
	iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
	iter_->SeekToFirst();
	for (iter_->SeekToFirst(); iter_->Valid(); iter_->Next()) counts_++;
	iter_->SeekToFirst();

	FILE * file = fopen(des.c_str(), "w");
	for(int i = 0; i < counts_; i ++)
	{
		Datum datum;
		datum.ParseFromString(iter_->value().ToString());
		string fileName = iter_->key().ToString();
		fprintf(file, "%s ", fileName.c_str());
		int len = LABEL_HEIGHT * LABEL_WIDTH * LABEL_LEN;
		for(int j = 0; j < len; j ++)
		{
			fprintf(file, "%f ", datum.label(j));
		}
		fprintf(file, "\n");
		iter_->Next();
	}
	fclose(file);

	return counts_;

}



int main(int argc, char** argv)
{
	string sourcefile(argv[1]);
	string labelFile(argv[2]);
	readLeveldb(sourcefile,labelFile);
	return 0;
}

















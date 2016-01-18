// 输出分割的结果

/*#include <cuda_runtime.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>*/

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

/*int readLeveldb(string &source, string &des)
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
		int len = LABEL_LEN * LABEL_LEN;
		for(int j = 0; j < len; j ++)
		{
			fprintf(file, "%d ", datum.label(j));
		}
		fprintf(file, "\n");
	}
	fclose(file);

	return counts_;

}*/

int CreateDir(const char *sPathName, int beg) {
	char DirName[256];
	strcpy(DirName, sPathName);
	int i, len = strlen(DirName);
	if (DirName[len - 1] != '/')
		strcat(DirName, "/");

	len = strlen(DirName);

	for (i = beg; i < len; i++) {
		if (DirName[i] == '/') {
			DirName[i] = 0;
			if (access(DirName, 0) != 0) {
				CHECK(mkdir(DirName, 0755) == 0)<< "Failed to create folder "<< sPathName;
			}
			DirName[i] = '/';
		}
	}

	return 0;
}

char buf[101000];
int main(int argc, char** argv)
{

	cudaSetDevice(0);
	Caffe::set_phase(Caffe::TEST);
	//Caffe::SetDevice(2);
	if (argc == 8 && strcmp(argv[7], "CPU") == 0) {
		LOG(ERROR) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	} else {
		LOG(ERROR) << "Using GPU";
		Caffe::set_mode(Caffe::GPU);
	}

	Caffe::set_mode(Caffe::GPU);
	Caffe::SetDevice(6);

	NetParameter test_net_param;
	ReadProtoFromTextFile(argv[1], &test_net_param);
	Net<float> caffe_test_net(test_net_param);
	NetParameter trained_net_param;
	ReadProtoFromBinaryFile(argv[2], &trained_net_param);
	caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

	vector<shared_ptr<Layer<float> > > layers = caffe_test_net.layers();
	//const DataLayer<float> *datalayer = dynamic_cast<const DataLayer<float>* >(layers[0].get());
	//CHECK(datalayer);

	string labelFile(argv[3]);
	int data_counts = 0;
	FILE * file = fopen(labelFile.c_str(), "r");
	while(fgets(buf,100000,file) > 0)
	{
		data_counts++;
	}
	fclose(file);

	vector<Blob<float>*> dummy_blob_input_vec;
	string rootfolder(argv[4]);
	rootfolder.append("/");
	CreateDir(rootfolder.c_str(), rootfolder.size() - 1);
	string folder;
	string fName;

	float output;
	int counts = 0;

	file = fopen(labelFile.c_str(), "r");

	Blob<float>* c1 = (*(caffe_test_net.bottom_vecs().rbegin()))[0];
    int c2 = c1->num();
	int batchCount = std::ceil(data_counts / (floor)(c2));//(test_net_param.layers(0).layer().batchsize()));//                (test_net_param.layers(0).layer().batchsize() ));

	string resulttxt = rootfolder + "poseResult.txt";
	FILE * resultfile = fopen(resulttxt.c_str(), "w");

	for (int batch_id = 0; batch_id < batchCount; ++batch_id)
	{
		LOG(INFO)<< "processing batch :" << batch_id+1 << "/" << batchCount <<"...";

		const vector<Blob<float>*>& result = caffe_test_net.Forward(dummy_blob_input_vec);
		Blob<float>* bboxs = (*(caffe_test_net.bottom_vecs().rbegin()))[0];
		int bsize = bboxs->num();

		const Blob<float>* labels = (*(caffe_test_net.bottom_vecs().rbegin()))[1];
		for (int i = 0; i < bsize && counts < data_counts; i++, counts++)
		{
			char fname[1010];
			int tnum;
			fscanf(file, "%s", fname);
			fscanf(file, "%d", &tnum);
			fprintf(resultfile, "%s ", fname);
			int channels = bboxs->channels();
			int height   = bboxs->height();
			int width    = bboxs->width();
			int len = channels * height * width;

			vector<pair<float,int> > pts;
			vector<pair<float,int> > pts2;
			for (int c = 0; c < channels; c ++)
			{
				pts.clear();
				pts2.clear();
				for(int h = 0; h < height; h ++)
					for(int w = 0; w < width; w ++)
					{
						//if((float)(bboxs->data_at(i, c, h, w)) < 0.8) continue;
						int id = h * width + w;
						pts.push_back(make_pair((float)(bboxs->data_at(i, c, h, w)), id ));

						//fprintf(resultfile, "%f ", (float)(bboxs->data_at(i, c, h, w)) );
					}
				sort(pts.begin(), pts.end());
				int maxid = pts[pts.size() - 1].second;
				int maxh = maxid / width;
				int maxw = maxid % width;

				for(int h = -3; h <= 3; h ++)
					for(int w = -3; w <=3 ;w ++)
					{
						int nowh = maxh + h;
						int noww = maxw + w;
						if(nowh >= 0 && nowh < height && noww >=0 && noww < width)
						{
							int id = nowh * width + noww;
							pts2.push_back(make_pair((float)(bboxs->data_at(i, c, nowh, noww)), id ));

						}
					}

				for(int k= 0; k < pts2.size(); k ++)
				{
					fprintf(resultfile, "%d %f ", pts2[k].second, pts2[k].first);
				}
				if(pts2.size() < 49)
				{
					for(int k = pts2.size(); k < 49; k ++)
					{
						fprintf(resultfile, "%d %f ", pts2[k].second, pts2[k].first);
					}
				}

			}
			fprintf(resultfile, "\n");
		}
	}

	fclose(resultfile);
	fclose(file);


	return 0;
}

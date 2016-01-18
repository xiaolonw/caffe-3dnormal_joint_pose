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
#define LABEL_LEN 3
#define LABEL_HEIGHT 48
#define LABEL_WIDTH 64




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

	Caffe::set_phase(Caffe::TEST);
	Caffe::SetDevice(0);
	//Caffe::set_mode(Caffe::CPU);

	if (argc == 8 && strcmp(argv[7], "CPU") == 0) {
		LOG(ERROR) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	} else {
		LOG(ERROR) << "Using GPU";
		Caffe::set_mode(Caffe::GPU);
	}


	//Caffe::set_mode(Caffe::CPU);

	NetParameter test_net_param;
	ReadProtoFromTextFile(argv[1], &test_net_param);
	Net<float> caffe_test_net(test_net_param);
	NetParameter trained_net_param;
	ReadProtoFromBinaryFile(argv[2], &trained_net_param);
	caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

	vector<shared_ptr<Layer<float> > > layers = caffe_test_net.layers();
	const DataLayer<float> *datalayer = dynamic_cast<const DataLayer<float>* >(layers[0].get());
	CHECK(datalayer);

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

	string resulttxt = rootfolder + "3dNormalResult.txt";
	FILE * resultfile = fopen(resulttxt.c_str(), "w");

	float * st = new float[LABEL_LEN * LABEL_HEIGHT * LABEL_WIDTH];

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
			fscanf(file, "%s", fname);
			for(int c = 0; c < LABEL_LEN; c ++)
			for(int w = 0; w < LABEL_WIDTH; w ++)
				for(int h = 0; h < LABEL_HEIGHT; h ++)
				{
					float lbl;
					fscanf(file,"%f", &lbl);
				}
			fprintf(resultfile, "%s ", fname);
			//char ss2[1010];
			//sprintf(ss2,"/home/dragon123/cnncode/3ddata/testReg/%d_fine_not.jpg",counts);
			//Mat imgOut(Size(LABEL_SIZE,LABEL_SIZE),CV_8UC3);
			//for(int c = 0; c < LABEL_LEN; c++)
			{
				for(int w = 0; w < LABEL_WIDTH; w ++)
					for(int h = 0; h < LABEL_HEIGHT; h ++)
					{
						float tnum1 = (float)(bboxs->data_at(i, 0, h, w));
						float tnum2 = (float)(bboxs->data_at(i, 1, h, w));
						float tnum3 = (float)(bboxs->data_at(i, 2, h, w));
						float z = tnum1 * tnum1 + tnum2 * tnum2 + tnum3 * tnum3;
						z = sqrt(z);
						st[w * LABEL_HEIGHT + h] = tnum1 / z;
						st[LABEL_HEIGHT * LABEL_WIDTH + w * LABEL_HEIGHT + h] = tnum2 / z;
						st[2 * LABEL_HEIGHT * LABEL_WIDTH + w * LABEL_HEIGHT + h] = tnum3 / z;
						//fprintf(resultfile, "%f ", tnum);
						//imgOut.at<cv::Vec3b>(h, w)[2 - c] = (uchar)(tnum) * 128 + 128);
					}
			}

			for(int c = 0; c < LABEL_LEN; c++)
			{
				for(int w = 0; w < LABEL_WIDTH; w ++)
					for(int h = 0; h < LABEL_HEIGHT; h ++)
					{
						float tnum = st[c * LABEL_HEIGHT * LABEL_WIDTH + w * LABEL_HEIGHT + h] ;
						fprintf(resultfile, "%f ", tnum);
			//			imgOut.at<cv::Vec3b>(h, w)[2 - c] = (uchar)((tnum) * 128 + 128);
					}
			}

			//imwrite(ss2,imgOut);
			fprintf(resultfile, "\n");
		}
	}

	delete st;

	fclose(resultfile);
	fclose(file);


	return 0;
}

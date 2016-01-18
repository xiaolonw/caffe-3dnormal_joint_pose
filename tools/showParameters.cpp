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
#define LABEL_SIZE 55



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

	NetParameter test_net_param;
	ReadProtoFromTextFile(argv[1], &test_net_param);
	Net<float> caffe_test_net(test_net_param);
	NetParameter trained_net_param;
	ReadProtoFromBinaryFile(argv[2], &trained_net_param);
	caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

	vector<shared_ptr<Layer<float> > > layers = caffe_test_net.layers();
	vector<shared_ptr<Blob<float> > >  blobs_  = layers[1]->blobs();
	int numOutput = blobs_[0]->num();
	int channels  = blobs_[0]->channels();
	int height  = blobs_[0]->height();
	int width  = blobs_[0]->width();

	FILE * file = fopen("/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n/firstParam.txt", "wb");
	int size = channels * width * height * numOutput;
	fwrite(&size, sizeof(int), 1, file);

	for(int c = 0 ; c < channels; c ++)
		for(int w = 0; w < width; w ++)
			for(int h = 0; h < height; h ++)
				for(int fn = 0; fn < numOutput; fn ++)
				{
					float tnum = (float)(blobs_[0]->data_at(fn,c,h,w));
					fwrite(&tnum, sizeof(float), 1, file);
				}

	fclose(file);




	return 0;
}

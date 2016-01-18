#include <cuda_runtime.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <iostream>
#include <cstring>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/solver.hpp"

using namespace caffe;
using namespace std;



int main(int argc, char** argv){
  if (argc < 4){
    printf("Not enough parameters.\n");
    return 1;
  }

	NetParameter test_net_param;
	ReadProtoFromTextFile(argv[1], &test_net_param);
	Net<float> caffe_test_net(test_net_param);

	for(int i = 2; i < 5; i ++)
	{
		NetParameter trained_net_param;
		ReadProtoFromBinaryFile(argv[i], &trained_net_param);
		caffe_test_net.CopyTrainedLayersFrom(trained_net_param);
	}

	NetParameter newnet_param;
	caffe_test_net.ToProto(&newnet_param);


	for(int i = 5; i < 8; i ++)
	{
		NetParameter trained_net_param;
		ReadProtoFromBinaryFile(argv[i - 3], &trained_net_param);
		Net<float> caffe_net_now(trained_net_param);
		caffe_net_now.CopyTrainedLayersFrom(newnet_param);
		NetParameter save_net_param;
		caffe_net_now.ToProto(&save_net_param);
		WriteProtoToBinaryFile(save_net_param, argv[i]);
	}

	return 0;
}

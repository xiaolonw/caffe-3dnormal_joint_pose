#!/usr/bin/env sh                                                                                                

# test_net_seg.bin test_proto pre_train_model label.txt outputfolder [CPU/GPU]

ROOTFILE=/home/dragon123/3dnormal_joint_cnncode/caffe-3dnormal_joint

GLOG_logtostderr=1  /home/dragon123/3dnormal_joint_cnncode/caffe-3dnormal_joint/build/tools/test_net_3dnormal.bin /home/dragon123/3dnormal_joint_cnncode/experiment/global_model/seg_test_2fc_3dnormal.prototxt /home/dragon123/3dnormal_joint_cnncode/experiment/global_model/3dnormal__iter_150000  /home/dragon123/3dnormal_joint_cnncode/experiment/globaldata/testLabels.txt  /home/dragon123/3dnormal_joint_cnncode/experiment/globaldata/results






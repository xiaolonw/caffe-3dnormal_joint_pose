#!/usr/bin/env sh                                                                                                

# test_net_seg.bin test_proto pre_train_model label.txt outputfolder [CPU/GPU]

ROOTFILE=/home/dragon123/3dnormal_joint_cnncode/caffe-3dnormal_joint

GLOG_logtostderr=1  /home/dragon123/3dnormal_joint_cnncode/caffe-3dnormal_joint/build/tools/test_net_3dnormal_win.bin /home/dragon123/3dnormal_joint_cnncode/experiment/local_copy/seg_test_2fc_3dnormal_fcn.prototxt /home/dragon123/3dnormal_joint_cnncode/experiment/local_copy/3dnormal_fcn  /home/dragon123/3dnormal_joint_cnncode/experiment/localdata/testLabels.txt   /home/dragon123/3dnormal_joint_cnncode/experiment/localdata/results_2






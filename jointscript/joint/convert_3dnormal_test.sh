#!/usr/bin/env sh

rootfolder=/home/dragon123/3dnormal_joint_cnncode/caffe-3dnormal_joint

GLOG_logtostderr=1 $rootfolder/build/examples/3dnormal/convert_normal_test.bin /home/dragon123/3dnormal_joint_cnncode/experiment/globaldata/croptest/ /home/dragon123/3dnormal_joint_cnncode/experiment/globaldata/testLabels.txt  /home/dragon123/3dnormal_joint_cnncode/experiment/globaldata/3d_test_db_small 0 0 55 55

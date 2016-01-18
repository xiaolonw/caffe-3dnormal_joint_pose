#!/usr/bin/env sh

rootfolder=/home/dragon123/3dnormal_joint_cnncode/caffe-3dnormal_joint

GLOG_logtostderr=1 $rootfolder/build/tools/demo_compute_image_mean.bin /home/dragon123/3dnormal_joint_cnncode/experiment/globaldata/3d_test_db_small_large  /home/dragon123/3dnormal_joint_cnncode/experiment/globaldata/3d_test_db_small_large/3d_mean.binaryproto

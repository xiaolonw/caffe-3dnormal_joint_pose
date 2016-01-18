#!/usr/bin/env sh

rootfolder=/home/dragon123/3dnormal_joint_cnncode/caffe-3dnormal_joint

GLOG_logtostderr=1 $rootfolder/build/examples/3dnormal/convert_normal.bin /home/dragon123/3dnormal_joint_cnncode/experiment/localdata/ /home/dragon123/3dnormal_joint_cnncode/experiment/localdata/bedroom_0041.txt  /home/dragon123/3dnormal_joint_cnncode/experiment/localdata/3d_train_db 0 0 55 55

#!/usr/bin/env sh

rootfolder=/home/dragon123/pose_cnncode/caffe-3dnormal_joint_pose

GLOG_logtostderr=1 $rootfolder/build/tools/demo_compute_image_mean.bin /home/dragon123/pose_cnncode/pose_train_db  /home/dragon123/pose_cnncode/pose_train_db/pose_mean.binaryproto

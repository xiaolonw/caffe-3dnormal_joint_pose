#!/usr/bin/env sh

rootfolder=/home/dragon123/pose_cnncode/caffe-3dnormal_joint_pose

GLOG_logtostderr=1 $rootfolder/build/examples/pose/convert_poseimg.bin /home/dragon123/cnncode/3ddata/croptest /home/dragon123/cnncode/3ddata/test3d.txt /home/dragon123/pose_cnncode/pose_train_db 0 0 240 240




#!/usr/bin/env sh

rootfolder=/home/dragon123/cnncode/caffe-3dnormal_r

GLOG_logtostderr=1 $rootfolder/build/tools/demo_compute_image_mean_float.bin $rootfolder/models/3d_normal_db/3d_train_db $rootfolder/models/3d_normal_db/3d_mean.binaryproto

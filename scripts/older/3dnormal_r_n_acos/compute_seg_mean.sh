#!/usr/bin/env sh

rootfolder=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r

GLOG_logtostderr=1 $rootfolder/build/tools/demo_compute_image_mean_float.bin $rootfolder/models/3d_normal_db_r/3d_train_db $rootfolder/models/3d_normal_db_r/3d_mean.binaryproto

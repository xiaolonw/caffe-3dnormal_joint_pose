#!/usr/bin/env sh

rootfolder=/N/u/xiaolonw/cnncode/caffe-3dnormal_r_n

GLOG_logtostderr=1 $rootfolder/build/tools/demo_compute_image_mean.bin /N/u/xiaolonw/cnncode/seg_cls/models/3dnormal_win_cls/3d_train_db  /N/u/xiaolonw/cnncode/seg_cls/models/3dnormal_win_cls/3d_train_db/3d_mean.binaryproto

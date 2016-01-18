#!/usr/bin/env sh

rootfolder=/N/u/xiaolonw/cnncode/caffe-3dnormal_r_n

GLOG_logtostderr=1 $rootfolder/build/examples/3dnormal/convert_normal.bin /N/u/xiaolonw/dataset/sliding_window/images/ /N/u/xiaolonw/cnncode/seg_cls/data/leveldb/win_cls/trainLabels.txt  /N/u/xiaolonw/cnncode/seg_cls/data/leveldb/win_cls/3d_train_db 0 1 55 55

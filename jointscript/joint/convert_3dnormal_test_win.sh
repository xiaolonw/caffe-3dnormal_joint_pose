#!/usr/bin/env sh

rootfolder=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_local

GLOG_logtostderr=1 $rootfolder/build/examples/3dnormal/convert_normal_test.bin /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window/images_test/ /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window/testLabels.txt  /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window/leveldb/3d_test_db_small 0 0 55 55

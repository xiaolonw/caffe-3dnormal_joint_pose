#!/usr/bin/env sh

rootfolder=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n

GLOG_logtostderr=1 $rootfolder/build_compute-0-5/examples/3dnormal/convert_normal_test.bin /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window/images_testontrain  /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window/testLabels_ontrain/trainLabels2.txt /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window/leveldb_testontrain/3d_train_db_2 0 0 55 55





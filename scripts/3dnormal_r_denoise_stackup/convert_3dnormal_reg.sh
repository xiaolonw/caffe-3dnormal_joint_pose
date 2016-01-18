#!/usr/bin/env sh

rootfolder=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r

GLOG_logtostderr=1 $rootfolder/build/examples/3dnormal/convert_normalReg.bin /nfs/hn46/dfouhey/deepProcessed/data/ /nfs/ladoga_no_backups/users/xiaolonw/3dnormal_data/train_test_cross/trainLabels.txt /nfs/ladoga_no_backups/users/xiaolonw/3dnormal_data/train_test_cross/testresults_cross/3d_train_db_reg 0 1 55 55 /nfs/ladoga_no_backups/users/xiaolonw/3dnormal_data/train_test_cross/testresults_cross/reg

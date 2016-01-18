#!/usr/bin/env sh

rootfolder=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n

GLOG_logtostderr=1 $rootfolder/build/examples/3dnormal/convert_normalReg_resize.bin /nfs/hn46/dfouhey/deepProcessed/data/ /nfs/ladoga_no_backups/users/xiaolonw/3dnormal_data/train_test_reg_edge/trainLabels.txt /nfs/ladoga_no_backups/users/xiaolonw/3dnormal_data/train_test_reg_resize/3d_train_db 0 1 64 48 /nfs/ladoga_no_backups/users/xiaolonw/3dnormal_data/train_test_cross/testresults/reg_resize

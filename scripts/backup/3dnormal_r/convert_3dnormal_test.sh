#!/usr/bin/env sh

rootfolder=/home/dragon123/cnncode/caffe-3dnormal_r

GLOG_logtostderr=1 $rootfolder/build/examples/3dnormal/convert_normalReg.bin /home/dragon123/cnncode/3ddata/croptest /home/dragon123/cnncode/3ddata/smallTestLabels.txt $rootfolder/models/3d_normal_db_r/3d_test_db_small 0 0 55 55 /home/dragon123/cnncode/3ddata/reg_test

#GLOG_logtostderr=1 $rootfolder/build/examples/3dnormal/convert_normal.bin /nfs/hn46/dfouhey/deepProcessed/data/ /nfs/hn46/xiaolonw/cnncode/viewer/train_test_3dnormal/testLabels.txt $rootfolder/models/3d_normal_db/3d_test_db 0 0 55 55

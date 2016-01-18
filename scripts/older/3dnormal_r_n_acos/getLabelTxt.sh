#!/usr/bin/env sh

rootfolder=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r

#GLOG_logtostderr=1 $rootfolder/build/tools/getLabelFromLeveldb.bin $rootfolder/models/3d_normal_db/3d_test_db /nfs/hn46/xiaolonw/cnncode/viewer/train_test_3dnormal/dbTestLabels.txt

GLOG_logtostderr=1 $rootfolder/build/tools/getLabelFromLeveldb.bin $rootfolder/models/3d_normal_db_r/3d_test_db_small /nfs/hn46/xiaolonw/cnncode/viewer/train_test_3dnormal_reg/dbTestLabels_small.txt

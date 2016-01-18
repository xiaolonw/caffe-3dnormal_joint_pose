#!/usr/bin/env sh

rootfolder=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n

#GLOG_logtostderr=1 $rootfolder/build/tools/getLabelFromLeveldb.bin $rootfolder/models/3d_normal_db/3d_test_db /nfs/hn46/xiaolonw/cnncode/viewer/train_test_3dnormal/dbTestLabels.txt

GLOG_logtostderr=1 $rootfolder/build/tools/getLabelFromLeveldb.bin /nfs/ladoga_no_backups/users/xiaolonw/3dnormal_data/train_test_reg_edge/3d_test_db_small /nfs/ladoga_no_backups/users/xiaolonw/3dnormal_data/train_test_reg_edge/3d_test_db_small/dbTestLabels_small.txt

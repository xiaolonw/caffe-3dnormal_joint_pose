#!/usr/bin/env sh

rootfolder=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n

GLOG_logtostderr=1 $rootfolder/build/tools/demo_compute_image_mean_float.bin /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/data/leveldb/3d_train_db_slv_tri_van /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/data/leveldb/3d_train_db_slv_tri_van/3d_mean.binaryproto

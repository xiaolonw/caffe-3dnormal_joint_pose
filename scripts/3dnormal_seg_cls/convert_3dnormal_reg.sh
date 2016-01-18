#!/usr/bin/env sh

rootfolder=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n

GLOG_logtostderr=1 $rootfolder/build/examples/3dnormal/convert_normal_seg.bin /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/data/imgPatches /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/data/leveldb/seg_cls/trainLabels.txt /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/data/leveldb/seg_cls/3d_train_db 0 1 55 55 /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/data/reg

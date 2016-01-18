#!/usr/bin/env sh

rootfolder=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n

GLOG_logtostderr=1 $rootfolder/build/examples/3dnormal/convert_normal_win_high_van.bin /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high/images/ /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high/trainLabels.txt  /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/data/leveldb/3d_train_db_slv_tri_van 0 1 55 55 /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high/reg_coarse_tri/ /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high/reg_local_tri/ /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high/reg_coarse_van_tri





#!/usr/bin/env sh

rootfolder=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n

GLOG_logtostderr=1 $rootfolder/build/examples/3dnormal/convert_normal_3in_test_scale.bin /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test_scale/images/ /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test_scale/locs/testLoc1.txt  /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test_scale/leveldb_van/3d_test_van_db1 0 0 55 55 /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test_scale/reg_coarse_tri /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test_scale/reg_local_tri  /home/xiaolonw/3ddata/sliding_window_edge_high_test_scale/reg_coarse_van_tri




#!/usr/bin/env sh

rootfolder=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n

GLOG_logtostderr=1 $rootfolder/build/examples/3dnormal/convert_normal_win_scale_test.bin /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test_scale/images/ /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test_scale/locs/testLoc2.txt  /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test_scale/leveldb/3d_test_db2 0 0 55 55 /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test_scale/reg_coarse_tri/ /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test_scale/reg_local_tri/

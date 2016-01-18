#!/usr/bin/env sh

rootfolder=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n

GLOG_logtostderr=1 $rootfolder/build/examples/3dnormal/convert_normal_win_high_test.bin /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test/images/ /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test/locs/testLoc1.txt  /home/xiaolonw/3ddata/3dnormal_win_cls_high/leveldb/3d_test_db1_2 0 0 55 55 /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test/reg_coarse_tri_2/ /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test/reg_local_tri/ 

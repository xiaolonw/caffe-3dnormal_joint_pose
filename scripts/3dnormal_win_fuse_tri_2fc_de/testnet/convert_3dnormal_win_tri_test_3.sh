#!/usr/bin/env sh

rootfolder=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n

GLOG_logtostderr=1 $rootfolder/build_onega/examples/3dnormal/convert_normal_win_high_test.bin /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test/images/ /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test/locs/testLoc3.txt  /nfs/hn38/users/xiaolonw/de_test_leveldb/3d_test_db3 0 0 55 55 /nfs/hn38/users/xiaolonw/de_test_patches/ /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test/reg_localedge_tri/ 

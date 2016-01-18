#!/usr/bin/env sh

rootfolder=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n

GLOG_logtostderr=1 $rootfolder/build/examples/3dnormal/convert_normal_3in_test_scale.bin /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test_scale/images/ /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test_scale/locs/testLoc3.txt  /home/xiaolonw/3ddata/sliding_window_edge_high_test_scale/leveldb/3d_test_edge2_db3 0 0 55 55 /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test_scale/reg_coarse_tri /home/xiaolonw/3ddata/sliding_window_edge_high_test_scale/reg_localedge_tri  /home/xiaolonw/3ddata/sliding_window_edge_high_test_scale/reg_edges




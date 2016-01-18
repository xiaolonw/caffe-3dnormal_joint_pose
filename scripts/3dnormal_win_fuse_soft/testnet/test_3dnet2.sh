#!/usr/bin/env sh                                                                                                

# test_net_seg.bin test_proto pre_train_model label.txt outputfolder [CPU/GPU]

ROOTFILE=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n

GLOG_logtostderr=1  /nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n/build/tools/test_net_3dnormal_high.bin /nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n/prototxt/3dnormal_win_fuse_soft_2fc/seg_test_2fc_3dnormal2.prototxt /home/xiaolonw/3ddata/3dnormal_win_cls_high/models/3d_train_slv_soft/3dnormal__iter_240000  /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window_edge_high_test/locs/testLoc2.txt /home/xiaolonw/3ddata/3dnormal_win_cls_high/3dfuseresult_55_soft/3d_fuse_2/




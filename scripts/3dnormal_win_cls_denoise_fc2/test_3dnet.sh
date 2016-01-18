#!/usr/bin/env sh                                                                                                

# test_net_seg.bin test_proto pre_train_model label.txt outputfolder [CPU/GPU]

ROOTFILE=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n

GLOG_logtostderr=1  /nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n/build/tools/test_net_3dnormal_win.bin /nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n/prototxt/3dnormal_win_cls_denoise_fc2/seg_test_2fc_3dnormal.prototxt /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window/models/3dnormal_win_cls_denoise_fc2/3dnormal__iter_390000  /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window/testLabels.txt   /nfs/ladoga_no_backups/users/xiaolonw/seg_cls/sliding_window/testresults/3dnormal_win_cls_denoise_fc2_39



#!/usr/bin/env sh                                                                                                

# test_net_seg.bin test_proto pre_train_model label.txt outputfolder [CPU/GPU]

ROOTFILE=/N/u/xiaolonw/cnncode/caffe-3dnormal_r_n

GLOG_logtostderr=1 $ROOTFILE/build/tools/test_net_3dnormal_win.bin $ROOTFILE/prototxt/3dnormal_win_cls/seg_test_2fc_3dnormal.prototxt /N/u/xiaolonw/cnncode/seg_cls/models/3dnormal_win_cls/3dnormal__iter_320000  /N/u/xiaolonw/cnncode/seg_cls/data/leveldb/win_cls/testLabels.txt   $ROOTFILE/3dnormal_result_small/3dnormal_win_cls


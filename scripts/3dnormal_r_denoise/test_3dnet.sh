#!/usr/bin/env sh                                                                                                

# test_net_seg.bin test_proto pre_train_model label.txt outputfolder [CPU/GPU]

ROOTFILE=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n

GLOG_logtostderr=1 $ROOTFILE/build/tools/test_net_3dnormal_reg.bin $ROOTFILE/prototxt/3dnormal_reg_denoise/seg_test_2fc_3dnormal.prototxt /nfs/ladoga_no_backups/users/xiaolonw/3dnormal_data/models_reg/3dnormal__iter_90000  /nfs/hn46/xiaolonw/cnncode/viewer/train_test_3dnormal_reg/dbTestLabels_small.txt   $ROOTFILE/3dnormal_result_small/denoise

#GLOG_logtostderr=1 $ROOTFILE/build/tools/test_net_3dnormal.bin $ROOTFILE/prototxt/3dnormal/seg_test_2fc_3dnormal.prototxt $ROOTFILE/models/3dnormal__iter_210000  /nfs/hn46/xiaolonw/cnncode/viewer/train_test_3dnormal/dbTestLabels.txt   $ROOTFILE/3dnormal_result

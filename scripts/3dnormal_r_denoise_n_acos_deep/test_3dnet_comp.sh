#!/usr/bin/env sh                                                                                                

# test_net_seg.bin test_proto pre_train_model label.txt outputfolder [CPU/GPU]

ROOTFILE=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n

GLOG_logtostderr=1 $ROOTFILE/build/tools/test_net_3dnormal_reg.bin $ROOTFILE/prototxt/3dnormal_reg_denoise_n_acos_deep/seg_test_2fc_3dnormal_comp.prototxt /nfs/ladoga_no_backups/users/xiaolonw/3dnormal_data/models_reg_n_acos_deep/3dnormal__iter_280000  /nfs/hn46/xiaolonw/cnncode/viewer/train_test_3dnormal_reg/dbTrainLabels_small.txt   $ROOTFILE/3dnormal_result_small/comp


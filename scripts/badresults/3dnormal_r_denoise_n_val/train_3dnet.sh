#!/usr/bin/env sh

TOOLS=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n/build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin /nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n/prototxt/3dnormal_reg_denoise_n_val/seg_solver_2fc_3dnormal.prototxt  


#!/usr/bin/env sh

TOOLS=/N/u/xiaolonw/cnncode/caffe-3dnormal_r_n/build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin /N/u/xiaolonw/cnncode/caffe-3dnormal_r_n/prototxt/3dnormal_win_cls_denoise_fc2/seg_solver_2fc_3dnormal.prototxt


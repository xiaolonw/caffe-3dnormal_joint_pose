#!/usr/bin/env sh

TOOLS=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n/build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin /nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n/prototxt/3dnormal_reg_denoise/seg_solver_2fc_3dnormal.prototxt /nfs/ladoga_no_backups/users/xiaolonw/3dnormal_data/models_reg/3dnormal__iter_50000.solverstate 


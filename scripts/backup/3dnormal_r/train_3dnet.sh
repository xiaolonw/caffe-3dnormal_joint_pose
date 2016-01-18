#!/usr/bin/env sh

TOOLS=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal/build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin /nfs/hn46/xiaolonw/cnncode/caffe-3dnormal/prototxt/3dnormal/seg_solver_2fc_3dnormal.prototxt


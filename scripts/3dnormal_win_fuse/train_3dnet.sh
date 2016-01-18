#!/usr/bin/env sh

TOOLS=/home/dragon123/cnncode/caffe-3dnormal_r/build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin /home/dragon123/cnncode/caffe-3dnormal_r/prototxt/3dnormal_win_fuse/seg_solver_2fc_3dnormal.prototxt

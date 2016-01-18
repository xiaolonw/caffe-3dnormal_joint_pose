#!/usr/bin/env sh

TOOLS=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n/build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin /nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n/prototxt/3dnormal_win_fuse_tri/seg_solver_2fc_3dnormal.prototxt /home/xiaolonw/3ddata/3dnormal_win_cls_high/models/3d_train_slv_tri/3dnormal__iter_180000.solverstate

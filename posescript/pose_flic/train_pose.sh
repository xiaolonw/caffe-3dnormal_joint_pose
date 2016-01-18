#!/usr/bin/env sh

TOOLS=/nfs/hn46/xiaolonw/pose_cnncode/caffe-3dnormal_joint_pose/build/tools

GLOG_logtostderr=1 $TOOLS/finetune_net.bin   /nfs/hn46/xiaolonw/pose_cnncode/caffe-3dnormal_joint_pose/posescript/pose_flic/pose_solver.prototxt /nfs/hn38/users/xiaolonw/pose_models/flic_coarse_1/pose__iter_18000






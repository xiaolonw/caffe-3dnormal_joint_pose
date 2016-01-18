#!/usr/bin/env sh

TOOLS=/nfs/hn46/xiaolonw/pose_cnncode/caffe-3dnormal_joint_pose2/build_compute-0-5/tools

GLOG_logtostderr=1 $TOOLS/finetune_net.bin   /nfs/hn46/xiaolonw/pose_cnncode/caffe-3dnormal_joint_pose2/posescript/pose_mpii_lecun/pose_solver.prototxt /nfs/hn38/users/xiaolonw/pose_models/mpii_coarse/pose__iter_25000






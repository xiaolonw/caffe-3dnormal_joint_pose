#!/usr/bin/env sh

TOOLS=/home/dragon123/pose_cnncode/caffe-3dnormal_joint_pose/build/tools

GLOG_logtostderr=1 $TOOLS/finetune_net.bin   /home/dragon123/pose_cnncode/caffe-3dnormal_joint_pose/posescript/pose_flic/pose_solver.prototxt /home/dragon123/pose_cnncode/models/pose__iter_1000

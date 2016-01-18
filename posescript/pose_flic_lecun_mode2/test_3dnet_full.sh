#!/usr/bin/env sh                                                                                                

# test_net_seg.bin test_proto pre_train_model label.txt outputfolder [CPU/GPU]

ROOTFILE=/home/dragon123/3dnormal_joint_cnncode/caffe-3dnormal_joint

GLOG_logtostderr=1  /home/dragon123/3dnormal_joint_cnncode/caffe-3dnormal_joint/build/tools/test_net_3dnormal_joint.bin /home/dragon123/3dnormal_joint_cnncode/experiment/joint_prototxt/seg_test_3dnormal_joint.prototxt /home/dragon123/3dnormal_joint_cnncode/experiment/joint_prototxt/3dnormal_global_full  /home/dragon123/3dnormal_joint_cnncode/experiment/joint_prototxt/3dnormal_local_full  /home/dragon123/3dnormal_joint_cnncode/experiment/joint_prototxt/3dnormal_fusion_full  /home/dragon123/3dnormal_joint_cnncode/experiment/joint_prototxt/test.txt /home/dragon123/3dnormal_joint_cnncode/experiment/joint_prototxt/3d_fuse


#!/usr/bin/env sh                                                                                                

# test_net_seg.bin test_proto pre_train_model label.txt outputfolder [CPU/GPU]

ROOTFILE=/home/dragon123/3dnormal_joint_cnncode/caffe-3dnormal_joint

GLOG_logtostderr=1  /home/dragon123/3dnormal_joint_cnncode/caffe-3dnormal_joint/build/tools/test_net_3dnormal_high.bin /home/dragon123/3dnormal_joint_cnncode/experiment/fusion_model/seg_test_fusion_single.prototxt /home/dragon123/3dnormal_joint_cnncode/experiment/joint_prototxt/3dnormal_fusion_full  /home/dragon123/3dnormal_joint_cnncode/experiment/fusion_model/testLoc.txt /home/dragon123/3dnormal_joint_cnncode/experiment/fusion_model/3d_fuse/



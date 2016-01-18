#!/usr/bin/env sh                                                                                                

# test_net_seg.bin test_proto pre_train_model label.txt outputfolder [CPU/GPU]

ROOTFILE=/nfs/hn46/xiaolonw/pose_cnncode/caffe-3dnormal_joint_pose

GLOG_logtostderr=1  /nfs/hn46/xiaolonw/pose_cnncode/caffe-3dnormal_joint_pose/build_compute-0-5/tools/test_net_pose.bin /nfs/hn46/xiaolonw/pose_cnncode/caffe-3dnormal_joint_pose/posescript/pose_flic_lecun_mode2/pose_test.prototxt /nfs/hn38/users/xiaolonw/pose_models/flic_coarse_lecun_mode2/pose__iter_20000 /nfs/hn38/users/xiaolonw/FLIC/data/testlist_256.txt /nfs/hn38/users/xiaolonw/FLIC/data/results_256_2


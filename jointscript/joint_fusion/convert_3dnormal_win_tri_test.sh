#!/usr/bin/env sh

rootfolder=/home/dragon123/3dnormal_joint_cnncode/caffe-3dnormal_joint

GLOG_logtostderr=1 $rootfolder/build/examples/3dnormal/convert_normal_win_high_test.bin /home/dragon123/3dnormal_joint_cnncode/experiment/fusion_model/images/ /home/dragon123/3dnormal_joint_cnncode/experiment/fusion_model/testLoc.txt  /home/dragon123/3dnormal_joint_cnncode/experiment/fusion_model/3d_test_db 0 0 55 55 /home/dragon123/3dnormal_joint_cnncode/experiment/fusion_model/reg_coarse_tri/ /home/dragon123/3dnormal_joint_cnncode/experiment/fusion_model/reg_local_tri/ 



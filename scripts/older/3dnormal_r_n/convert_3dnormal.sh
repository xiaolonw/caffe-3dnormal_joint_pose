#!/usr/bin/env sh

rootfolder=/home/cnncode/caffe-3dnormal_r

GLOG_logtostderr=1 $rootfolder/build/examples/3dnormal/convert_normalReg.bin /home/dragon123/cnncode/3ddata/data/ /home/dragon123/cnncode/3ddata/smallTestLabels.txt $rootfolder/models/3d_normal_db/3d_train_db 0 1 55 55 /home/dragon123/cnncode/3ddata/reg


#/home/dragon123/cnncode/3ddata/data/ /home/dragon123/cnncode/3ddata/smallTrainLabels.txt /home/cnncode/caffe-3dnormal_r/models/3d_normal_db/3d_train_db 0 1 55 55 /home/dragon123/cnncode/3ddata/reg

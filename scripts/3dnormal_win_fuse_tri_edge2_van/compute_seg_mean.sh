#!/usr/bin/env sh

rootfolder=/nfs/hn46/xiaolonw/cnncode/caffe-3dnormal_r_n

GLOG_logtostderr=1 $rootfolder/build/tools/demo_compute_image_mean_float.bin /home/xiaolonw/3ddata/3dnormal_win_cls_high/leveldb_fusetrain/3d_train_db_slv_tri_edge2_van /home/xiaolonw/3ddata/3dnormal_win_cls_high/leveldb_fusetrain/3d_train_db_slv_tri_edge2_van/3d_mean.binaryproto

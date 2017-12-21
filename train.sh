#!/bin/sh
caffe.bin train -solver=model/VGGSolver.prototxt -weights=/home/users/duanqi01/PathData/SJPathData/ResultData1/train/pretrained/VGG_full_conv.caffemodel -gpu=0,1 >&1 | tee log/caffe-train.log
echo 'Done!'


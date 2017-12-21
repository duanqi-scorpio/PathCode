#!/bin/sh
#caffe.bin train -solver=model/VGGSolver.prototxt -weights=/home/fengyifan/medical/pathology/SJTU_PROJ/Experiments3_bk/train/pretrained/VGG_full_conv.caffemodel -gpu=0 >&1 | tee log/caffe-train.log
#cd /home/users/duanqi01/caffe-for-kongming2/for_kongming
#source set_env.sh
#echo 'Set Caffe Environment Done!'
cd /home/users/duanqi01/RJPath-master_V1
#LD_LIBRARY_PATH=/opt/compiler/gcc-4.8.2/lib:$LD_LIBRARY_PATH /opt/compiler/gcc-4.8.2/lib64/ld-linux-x86-64.so.2  /home/users/duanqi01/caffe-for-kongming2/for_kongming/third/python2.7/bin/python random_divide_data.py
python random_divide_data.py
echo 'Excute random_divide_data.py Done!'
#LD_LIBRARY_PATH=/opt/compiler/gcc-4.8.2/lib:$LD_LIBRARY_PATH /opt/compiler/gcc-4.8.2/lib64/ld-linux-x86-64.so.2  /home/users/duanqi01/caffe-for-kongming2/for_kongming/third/python2.7/bin/python preprocess.py
python preprocess.py
echo 'Excute preprocess.py Done!'
#LD_LIBRARY_PATH=/opt/compiler/gcc-4.8.2/lib:$LD_LIBRARY_PATH /opt/compiler/gcc-4.8.2/lib64/ld-linux-x86-64.so.2  /home/users/duanqi01/caffe-for-kongming2/for_kongming/distribute/bin/caffe.bin train -solver=model/VGGSolver.prototxt -weights=/home/users/duanqi01/PathData/RJPathData/ResultData1/train/pretrained/VGG_full_conv.caffemodel -gpu=0,1,2,3 >&1 | tee log/caffe-train.log
echo 'Done!'


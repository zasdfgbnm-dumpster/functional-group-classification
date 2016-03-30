#!/bin/bash

label='labelC2O'

#generate network and data file list for cross validation
for cv in {0..2};do
	sed "s/<cv>/$cv/g" solver.prototxt > solver-$cv.prototxt
	sed "s/<label>/$label/g;s/<cv>/$cv/g" irnet.prototxt > irnet-$cv.prototxt
	echo "data-$cv.h5" > test-$cv.txt
	echo data-{0..2}.h5|sed 's/ /\n/g'|grep -v $cv > train-$cv.txt
done

#train the three net
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.0/lib64
for cv in {0..2};do
	/apps/caffe/build/tools/caffe train --solver=solver-$cv.prototxt 2>&1|tee train-$cv.log
done

#analyze the result
./analyze_result.py

#clean up
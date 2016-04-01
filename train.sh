#!/bin/bash

label=$(cat label)
net=$(cat net)

#generate network and data file list for cross validation
for cv in {0..2};do
	sed "s/<cv>/$cv/g" $net-solver.prototxt > $net-solver-$cv.prototxt
	sed "s/<label>/$label/g;s/<cv>/$cv/g" $net.prototxt > $net-$cv.prototxt
	echo "data-$cv.h5" > test-$cv.txt
	echo data-{0..2}.h5|sed 's/ /\n/g'|grep -v $cv > train-$cv.txt
done

#train the three net
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.0/lib64:/apps/caffe/build/lib
for cv in {0..2};do
	/apps/caffe/build/tools/caffe train --solver=$net-solver-$cv.prototxt 2>&1|tee $net-train-$cv.log
done

#analyze the result
./analyze_result.py 2>/dev/null

#clean up

#!/bin/bash
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.0/lib64 /apps/caffe/build/tools/caffe train --solver=solver.prototxt 2>&1|tee train.log

#!/usr/bin/python2
import sys
import caffe
import numpy

files = sys.argv[1:]

for i in files:
    h = numpy.load(i)
    blob = caffe.io.array_to_blobproto( h.reshape( (1,1,2,2) ) )
    fn = '.'.join([i.split('.')[0],'binaryproto'])
    with open(fn,'wb') as f :
        f.write( blob.SerializeToString() )

#!/usr/bin/python2
import caffe
import h5py
import sys

# modelfn = 'irnet.prototxt'
# modelwfn = 'iter_35000.caffemodel'

dbsize = 3000
batchsize = 60

def run_test(net,weight,testdbfn,label):
	net = caffe.Net(net,weight,caffe.TEST)
	testdb = h5py.File(testdbfn)
	totcountpp = 0
	totcountpn = 0
	totcountnp = 0
	totcountnn = 0
	totcount = 0
	for i in range(dbsize/batchsize):
		batch_data = testdb['data'][i*batchsize:(i+1)*batchsize]
		batch_label_raw = testdb[label][i*batchsize:(i+1)*batchsize]
		batch_label = [int(e[0][0][0]) for e in batch_label_raw]
		net.blobs['data'].data[...] = batch_data
		net.forward()
		predicted_label_raw = net.blobs['ip2'].data
		predicted_label = [ 0 if x[0]>x[1] else 1 for x in predicted_label_raw]
		countpp = 0
		countpn = 0
		countnp = 0
		countnn = 0
		count = 0
		for e,p in zip(batch_label,predicted_label):
			# print 'batch:',i,'\texpected:',e,'\tgot:',p
			count += 1
			if e and p:
				countpp += 1
				continue
			if e and not p:
				countpn += 1
				continue
			if not e and p:
				countnp += 1
				continue
			if not e and not p:
				countnn += 1
				continue
		totcountpp += countpp
		totcountpn += countpn
		totcountnp += countnp
		totcountnn += countnn
		totcount += count
		overall_accuracy   = 1.0 * (countpp+countnn)/count
		paccuracy = 1.0 * countpp/(countpp+countpn)
		naccuracy = 1.0 * countnn/(countnp+countnn)
		print >>sys.stderr, 'batch number:',i
		print >>sys.stderr, 'overall accuracy:', 100 * overall_accuracy,'%'
		print >>sys.stderr, '+ accuracy:', 100 * paccuracy,'%'
		print >>sys.stderr, '- accuracy:', 100 * naccuracy,'%'
	return overall_accuracy,paccuracy,naccuracy

if __name__ == "__main__":
	import sys
	overall,paccuracy,naccuracy = run_test(sys.argv[1],sys.argv[2],sys.argv[3],open('label').readline().strip())
	print 'overall accuracy:', 100 * overall,'%'
	print '+ accuracy:', 100 * paccuracy,'%'
	print '- accuracy:', 100 * naccuracy,'%'

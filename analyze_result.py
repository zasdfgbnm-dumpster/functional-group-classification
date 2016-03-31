#!/usr/bin/python2

from test import run_test

overall_accuracy_arr = []
paccuracy_arr = []
naccuracy_arr = []

for i in range(3):
	net = 'irnet-{}.prototxt'.format(i)
	weight = '{}_iter_35000.caffemodel'.format(i)
	testdbfn = 'data-{}.h5'.format(i)
	overall_accuracy,paccuracy,naccuracy = run_test(net,weight,testdbfn)
	print 'cross validation:',i,'\toverall accuracy:',overall_accuracy,'\t++ rate:', paccuracy,'\t-- rate:', naccuracy
	overall_accuracy_arr.append(overall_accuracy)
	paccuracy_arr.append(paccuracy)
	naccuracy.append(naccuracy)
	
print 'Summary:'
print 'overall accuracy:',sum(overall_accuracy_arr)/len(overall_accuracy_arr)
print '++ rate:',sum(paccuracy_arr)/len(paccuracy_arr)
print '-- rate:',sum(naccuracy_arr)/len(naccuracy_arr)

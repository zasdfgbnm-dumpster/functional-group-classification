#!/usr/bin/python2

from test import run_test

overall_accuracy_arr = []
paccuracy_arr = []
naccuracy_arr = []
label=open('label').readline().strip()
net=open('net').readline().strip()

for i in range(3):
	netfn = '{}-{}.prototxt'.format(net,i)
	weightfn = '{}_{}_iter_35000.caffemodel'.format(net,i)
	testdbfn = 'data-{}.h5'.format(i)
	overall_accuracy,paccuracy,naccuracy = run_test(netfn,weightfn,testdbfn,label)
	print 'cross validation:',i,'\toverall accuracy:',overall_accuracy,'\t++ rate:', paccuracy,'\t-- rate:', naccuracy
	overall_accuracy_arr.append(overall_accuracy)
	paccuracy_arr.append(paccuracy)
	naccuracy_arr.append(naccuracy)

print '\nSummary:'
print 'overall accuracy:',sum(overall_accuracy_arr)/len(overall_accuracy_arr)
print '++ rate:',sum(paccuracy_arr)/len(paccuracy_arr)
print '-- rate:',sum(naccuracy_arr)/len(naccuracy_arr)

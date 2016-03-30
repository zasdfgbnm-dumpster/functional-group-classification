#!/usr/bin/python

import h5py
from feature_extractor import smiles_searcher,or_extractor
from random import shuffle
import numpy

num_train = 10000
num_test = 2000
dimir = 759

extractors = [
	smiles_searcher('C2O','C=O'),
	smiles_searcher('CO','CO'),
	smiles_searcher('CN','CN'),
	smiles_searcher('C2C','C=C'),
	smiles_searcher('CC2OC','CC(=O)C'),
	or_extractor('NOorN2O',smiles_searcher('','NO'),smiles_searcher('','N=O'))
]

# load db to memory and calculate features
fdb = open('/home/gaoxiang/MEGA/step18/db-18-shuf.txt')
irdb = []
counts = [ 0 for x in extractors ]
for i in fdb:
	l = i.strip().split()
	myid = l[0]
	ir = [ float(x) for x in l[1:] ]
	features = [ x.extract(myid) for x in extractors ]
	counts = [ x+int(y) for x,y in zip(counts,features) ]
	irdb.append((myid,ir,features))

# keep only num_test+num_train items with highest enabled bits in features
irdb.sort(key=lambda x : sum(x[2]),reverse=True)
irdb = irdb[:num_train+num_test]
shuffle(irdb)

# do some statistics
print('Statistics:')
counts = [ 0 for x in extractors ]
for x,y,features in irdb:
	counts = [ x+int(y) for x,y in zip(counts,features) ]
for x,y in zip(extractors,counts):
	print( x.name, y, 1.0*y/(num_test+num_train) )

# write database
trainf = h5py.File('train.h5','w')
testf = h5py.File('test.h5','w')
offset = 0
for f,n in [(trainf,num_train),(testf,num_test)]:
	# dimensions are (periodNum,channel,width,height)
	f.create_dataset('data',(n,1,1,dimir),dtype='f8')
	f.create_dataset('nist_id',(n,),dtype=h5py.special_dtype(vlen=bytes))
	for (myid,ir,features),index in zip(irdb[offset:offset+n],range(n)):
		f['data'][index] = numpy.array(ir).reshape((1,1,dimir))
		f['nist_id'][index] = myid.encode('ascii')
	for e,exidx in zip(extractors,range(len(extractors))):
		label_name = 'label'+e.name
		f.create_dataset(label_name,(n,1,1,1),dtype='f4')
		for (myid,ir,features),index in zip(irdb[offset:offset+n],range(n)):
			f[label_name][index] = numpy.array(features[exidx]).reshape((1,1,1))
	offset += n

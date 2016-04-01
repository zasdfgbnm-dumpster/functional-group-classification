#!/usr/bin/python

import h5py
from feature_extractor import smiles_searcher,or_extractor
from random import shuffle
import numpy

num_total = 12000
num_split = 3000
num_pieces = int(num_total/num_split)
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
irdb = irdb[:num_total]
shuffle(irdb)

# do some statistics
print('Statistics:')
counts = [ 0 for x in extractors ]
for x,y,features in irdb:
	counts = [ x+int(y) for x,y in zip(counts,features) ]
for x,y in zip(extractors,counts):
	print( x.name, y, 1.0*y/(num_total) )

# write H for infogain loss
for e,y in zip(extractors,counts):
	h = numpy.eye(2,dtype='f4')
	h[0][0] = 1.0*y/num_total
	h[1][1] = 1.0*(num_total-y)/num_total
	numpy.save('H_label{}'.format(e.name),h)

# write database
offset = 0
for i in range(num_pieces):
	# dimensions are (periodNum,channel,width,height)
	f = h5py.File('data-{}.h5'.format(i),'w')
	f.create_dataset('data',(num_split,1,1,dimir),dtype='f8')
	f.create_dataset('nist_id',(num_split,),dtype=h5py.special_dtype(vlen=bytes))
	for (myid,ir,features),index in zip(irdb[offset:offset+num_split],range(num_split)):
		f['data'][index] = numpy.array(ir).reshape((1,1,dimir))
		f['nist_id'][index] = myid.encode('ascii')
	for e,exidx in zip(extractors,range(len(extractors))):
		label_name = 'label'+e.name
		f.create_dataset(label_name,(num_split,1,1,1),dtype='f4')
		for (myid,ir,features),index in zip(irdb[offset:offset+num_split],range(num_split)):
			f[label_name][index] = numpy.array(features[exidx]).reshape((1,1,1))
	offset += num_split

from rdkit import Chem

def nist_id_to_rdkit_mol(myid):
	path = '/home/gaoxiang/MEGA/shared/mol_files/{}.mol'
	return Chem.MolFromMolFile(path.format(myid))

class extractor(object):
	def __init__(self,name):
		self.name = name
	def extract(self,myid):
		pass

class or_extractor(extractor):
	def __init__(self,name,*extractors):
		super().__init__(name)
		self.extractors = extractors
	def extract(self,myid):
		inside = False
		for i in self.extractors:
			inside = inside or i.extract(myid)
		return inside

class and_extractor(extractor):
	def __init__(self,name,*extractors):
		super().__init__(name)
		self.extractors = extractors
	def extract(self,myid):
		inside = True
		for i in self.extractors:
			inside = inside and i.extract(myid)
		return inside

class substruct_searcher(extractor):
	def __init__(self,name,substructure):
		super().__init__(name)
		self.substructure = substructure
	def extract(self,myid):
		m = nist_id_to_rdkit_mol(myid)
		return m.HasSubstructMatch(self.substructure)

class smiles_searcher(substruct_searcher):
	def __init__(self,name,smiles):
		super().__init__(name,Chem.MolFromSmiles(smiles))

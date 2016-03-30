from rdkit import Chem

def nist_id_to_rdkit_mol(myid):
    path = '/home/gaoxiang/MEGA/shared/mol_files/{}.mol'
    return Chem.MolFromMolFile(path.format(myid))

class extractor(object):
	def __init__(self):
		pass
	def extract(self,myid):
		pass

class or_extractor(extractor):
	def __init__(self,*extractors):
		super.__init__()
		self.extractors = extractors
	def extract(self,myid):
		inside = False
		for i in self.extractors:
			inside = inside or i.extract(myid)

class and_extractor(extractor):
	def __init__(self,*extractors):
		super.__init__()
		self.extractors = extractors
	def extract(self,myid):
		inside = True
		for i in self.extractors:
			inside = inside and i.extract(myid)

class substruct_searcher(extractor):
    def __init__(self,substructure):
        super.__init__()
        self.substructure = substructure
    def extract(self,myid):
        m = nist_id_to_rdkit_mol(myid)
        return m.HasSubstructMatch(self.substructure)

class smiles_searcher(substruct_searcher):
    def __init__(self,smiles):
        super.__init__(Chem.MolFromSmiles(smiles))

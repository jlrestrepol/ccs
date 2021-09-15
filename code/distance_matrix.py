import Bio.PDB
import numpy as np
import os

pdb_id = "ranked_0"
pdb_filename = "./predictions/reduced_dbs/test7/ranked_0.pdb"

#print(os.getcwd())
structure = Bio.PDB.PDBParser().get_structure(pdb_id, pdb_filename)

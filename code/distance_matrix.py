import Bio.PDB
import numpy as np
import os

pdb_id = "ranked_0"
pdb_filename = "/fs/pool/pool-cox-projects-fold/predictions/reduced_dbs/test_roseta/ranked_0.pdb"

structure = Bio.PDB.PDBParser().get_structure(pdb_id, pdb_filename)

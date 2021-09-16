#%%
import Bio.PDB
import numpy
import matplotlib.pyplot as plt 
#%%
#pdb_id = "ranked_0"
#pdb_filename = "/fs/pool/pool-cox-projects-fold/predictions/reduced_dbs/test_roseta/ranked_0.pdb"

structure_alpha = Bio.PDB.PDBParser().get_structure("alpha","../data/prediction_alphafold.pdb")
structure_rosetta = Bio.PDB.PDBParser().get_structure("rosetta","../data/prediction_rosetta.pdb")


model_alpha = structure_alpha[0]
model_rosetta = structure_rosetta[0]
#%%
def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return numpy.sqrt(numpy.sum(diff_vector * diff_vector))

def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = numpy.zeros((len(chain_one), len(chain_two)), numpy.float)
    for row, residue_one in enumerate(chain_one) :
        for col, residue_two in enumerate(chain_two) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer

def are_alligned(chain_one, chain_two):
    """Verifies that both chains have the same residues in the same order"""
    res_list_one = numpy.fromiter( (res.get_resname() for res in chain_one.get_residues()), 
    dtype = 'S128', count = -1 )
    res_list_two = numpy.fromiter( (res.get_resname() for res in chain_two.get_residues()), 
    dtype = 'S128', count = -1 )
    return numpy.array_equal(res_list_one, res_list_two)

#%%
print("Are the two chains alligned?", are_alligned(model_alpha["A"], model_rosetta["A"]))

dist_matrix_alpha = calc_dist_matrix(model_alpha["A"], model_alpha["A"])
dist_matrix_rosetta = calc_dist_matrix(model_rosetta["A"], model_rosetta["A"])

#contact_map = dist_matrix < 12.0

res_list_alpha = numpy.fromiter((res.get_resname() for res in model_alpha.get_residues()), 
dtype = 'S128', count = -1 )

res_list_rosetta = numpy.fromiter((res.get_resname() for res in model_rosetta.get_residues()), 
dtype = 'S128', count = -1 )

print("Minimum distance", numpy.min(dist_matrix))
print("Maximum distance", numpy.max(dist_matrix))

fig = plt.gcf()
fig.set_size_inches(28, 10)
plt.imshow(numpy.transpose(dist_matrix))
plt.colorbar()
plt.title("Distance Matrix")
plt.xlabel("Alpha Fold")
plt.ylabel("Alpha Fold")
plt.xticks(numpy.arange(res_list_one.size), res_list_one.astype('str'), rotation = 90)
plt.yticks(numpy.arange(res_list_two.size), res_list_two.astype('str'), rotation = 0)
plt.show()
# %%

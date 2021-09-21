#%%
import Bio.PDB
import numpy
import matplotlib.pyplot as plt 
from itertools import chain
import scipy.spatial
#%%
def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return numpy.sqrt(numpy.sum(diff_vector * diff_vector))

def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = numpy.zeros((len(chain_one), len(chain_two)), numpy.float)
    for row, residue_one in enumerate(chain_one):
        for col, residue_two in enumerate(chain_two) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer

def calc_dist_matrix_numpy(chain_one, chain_two):
    """Efficient implementation of distance matrix calculation"""
    coords_one = numpy.fromiter( chain.from_iterable(res["CA"].coord for res in chain_one), dtype = 'f', count = -1)
    coords_two = numpy.fromiter( chain.from_iterable(res["CA"].coord for res in chain_two), dtype = 'f', count = -1)
    coords_one_3d = coords_one.reshape((-1, 3))
    coords_two_3d = coords_two.reshape((-1, 3))
    distance_matrix = scipy.spatial.distance_matrix(coords_one_3d, coords_two_3d)
    return distance_matrix 

def are_alligned(chain_one, chain_two):
    """Verifies that both chains have the same residues in the same order"""
    res_list_one = numpy.fromiter( (res.get_resname() for res in chain_one.get_residues()), 
    dtype = 'S128', count = -1 )
    res_list_two = numpy.fromiter( (res.get_resname() for res in chain_two.get_residues()), 
    dtype = 'S128', count = -1 )
    return numpy.array_equal(res_list_one, res_list_two)
#%%

def main():
    structure_alpha = Bio.PDB.PDBParser().get_structure("alpha","../Data/prediction_alphafold.pdb")
    structure_rosetta = Bio.PDB.PDBParser().get_structure("rosetta","../Data/prediction_rosetta.pdb")


    model_alpha = structure_alpha[0]
    model_rosetta = structure_rosetta[0]
    print("Are the two chains alligned?", are_alligned(model_alpha["A"], model_rosetta["A"]))

    dist_matrix_alpha = calc_dist_matrix(model_alpha["A"], model_alpha["A"])
    dist_matrix_rosetta = calc_dist_matrix(model_rosetta["A"], model_rosetta["A"])

    res_list_alpha = numpy.fromiter((res.get_resname() for res in model_alpha.get_residues()), 
    dtype = 'S128', count = -1 )

    res_list_rosetta = numpy.fromiter((res.get_resname() for res in model_rosetta.get_residues()), 
    dtype = 'S128', count = -1 )

    print("Minimum distance", numpy.min(dist_matrix_alpha))
    print("Maximum distance", numpy.max(dist_matrix_alpha))

    fig = plt.gcf()
    fig.set_size_inches(28, 10)
    plt.imshow(numpy.transpose(dist_matrix_alpha))
    plt.colorbar()
    plt.title("Distance Matrix")
    plt.xlabel("Alpha Fold")
    plt.ylabel("Alpha Fold")
    plt.xticks(numpy.arange(res_list_alpha.size), res_list_alpha.astype('str'), rotation = 90)
    plt.yticks(numpy.arange(res_list_alpha.size), res_list_alpha.astype('str'), rotation = 0)
    plt.show()

    #Get sequences and possibly even peptides
    ppb = Bio.PDB.PPBuilder()

    peptide_list = []
    for pp in ppb.build_peptides(structure_alpha):
        peptide_list.append(pp.get_sequence())

#%%

if __name__ == "__main__":
    main()
# %%

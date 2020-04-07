import json
from ase import Atoms, Atom
from ase.visualize import view
from ase.constraints import dict2constraint
from ase.calculators.singlepoint import SinglePointCalculator

def make_atoms_from_doc(doc):
    '''
    This is the inversion function for `make_doc_from_atoms`; it takes
    Mongo documents created by that function and turns them back into
    an ase.Atoms object.

    Args:
        doc     Dictionary/json/Mongo document created by the
                `make_doc_from_atoms` function.
    Returns:
        atoms   ase.Atoms object with an ase.SinglePointCalculator attached
    '''
    atoms = Atoms([Atom(atom['symbol'],
                        atom['position'],
                        tag=atom['tag'],
                        momentum=atom['momentum'],
                        magmom=atom['magmom'],
                        charge=atom['charge'])
                   for atom in doc['atoms']['atoms']],
                  cell=doc['atoms']['cell'],
                  pbc=doc['atoms']['pbc'],
                  info=doc['atoms']['info'],
                  constraint=[dict2constraint(constraint_dict)
                              for constraint_dict in doc['atoms']['constraints']])
    results = doc['results']
    calc = SinglePointCalculator(energy=results.get('energy', None),
                                 forces=results.get('forces', None),
                                 stress=results.get('stress', None),
                                 atoms=atoms)
    atoms.set_calculator(calc)
    return atoms
    
CovalentRadius = {'Ag':1.46,
 'Al':1.11,
 'As':1.21,
 'Au':1.21,
 'Bi':1.46,
 'C':0.77,
 'Ca':1.66,
 'Cd':1.41,
 'Co':1.21,
 'Cr':1.26,
 'Cu':1.21,
 'Cs':2.46,
 'Fe':1.26,
 'Ga':1.16,
 'Ge':1.22,
 'H':0.37,
 'Hf':1.41,
 'In':1.41,
 'Ir':1.21,
 'K':2.06,
 'Mn':1.26,
 'Mo':1.31,
 'N':0.74,
 'Na':1.66,
 'Nb':1.31,
 'Ni':1.21,
 'O':0.74,
 'Os':1.16,
 'P':1.1,
 'Pb':1.66,
 'Pd':1.26,
 'Pt':1.21,
 'Re':1.21,
 'Rh':1.21,
 'Ru':1.16,
 'S':1.04,
 'Sb':1.41,
 'Sc':1.46,
 'Se':1.17,
 'Si':1.17,
 'Sn':1.4,
 'Sr':1.86,
 'Ta':1.31,
 'Tc':1.21,
 'Te':1.37,
 'Ti':1.26,
 'V':1.21,
 'W':1.21,
 'Y':1.66,
 'Zn':1.21,
 'Zr':1.41}
 
pbcs = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [1, 1, 0], [-1, 1, 0], [0, -1, 0], [1, -1, 0], [-1, -1, 0], [0, 0, 1], [1, 0, 1], [-1, 0, 1], [0, 1, 1], [1, 1, 1], [-1, 1, 1], [0, -1, 1], [1, -1, 1], [-1, -1, 1], [0, 0, -1], [1, 0, -1], [-1, 0, -1], [0, 1, -1], [1, 1, -1], [-1, 1, -1], [0, -1, -1], [1, -1, -1], [-1, -1, -1]]
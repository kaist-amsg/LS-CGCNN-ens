from Util import make_atoms_from_doc, CovalentRadius, pbcs
import pickle
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cdist
from ase.data import chemical_symbols
import pickle
import json
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
import os 
import csv
from ase.io import write
mat = StructureMatcher(primitive_cell=False)

def ProcessData(data,adsan,upperbound,path,radius=12.0,nnbr=9):
    ## find unused element for faking
    real = set()
    for datum in data:
        syms = []
        for a in datum['atoms']['atoms']:
            if a['tag'] == 0:
                syms.append(a['symbol'])
        real |= set(syms)

    c = chemical_symbols[1:100]
    for r in real:
        c.remove(r)
    
    fakemap = {}
    for i,r in enumerate(real):
        fakemap[chemical_symbols.index(r)] = chemical_symbols.index(c[i])
    json.dump(fakemap,open(path+'_sub_map.json','w'))
    
    datainitsurf = []
    for datum in tqdm(data):
        if datum['adsorption_energy'] < -3.0 or datum['adsorption_energy'] > 1.0:
            continue
        ## Find binding site atoms
        ### Get atoms
        ai = make_atoms_from_doc(datum['initial_configuration'])
        a  = make_atoms_from_doc(datum)
        ### Adsorbate index
        adsidx = np.where(a.get_tags()==1)[0]
        ### surface index
        surfidx = np.where(a.get_tags()==0)[0]
        surfmap = {i:surfidx[i] for i in range(len(surfidx))}
        an = a.get_atomic_numbers()
        sym = a.get_chemical_symbols()
        ### carbon
        Cidx = adsidx[an[adsidx]==adsan][0]
        ### positions
        CPos = a.get_positions()[Cidx,:]
        SurfPos = a.get_scaled_positions(wrap=False)[surfidx]
        SurfPosPbcs = np.concatenate(np.dot(SurfPos[None,:,:]+np.array(pbcs)[:,None,:],a.cell))
        ### find dist
        dists = cdist([CPos],SurfPosPbcs)[0]
        ### get criteria
        criteria =[]
        for _ in range(len(pbcs)):
            for i in surfmap:
                criteria.append(CovalentRadius[sym[surfmap[i]]]+CovalentRadius['C'])
        criteria = np.array(criteria)*upperbound
        ### apply criteria
        BindingSiteAtomIdx = [surfmap[i] for i in np.remainder(np.where(dists<criteria)[0],len(surfmap))] # map back to original
        if len(BindingSiteAtomIdx) == 0:
            continue # no binding site.
        ### Apply tag for binding site atoms
        tags = ai.get_tags()
        tags[BindingSiteAtomIdx] = 2 # 0 Surface atom, 1 adsorbate atom, 2 labeled surface atom
        ai.set_tags(tags)
        ## Apply Labeling
        ### Relaxed surface Input
        relsurf = ai.copy()
        del relsurf[np.where(relsurf.get_tags()==1)[0]]
        tags = relsurf.get_tags()
        tags[tags==2] = 1
        relsurf.set_tags(tags)
        ### initial surface input
        #### first find mapping to slab information
        sf = make_atoms_from_doc(datum['slab_information']) 
        sfpbcpos = np.concatenate(np.dot(sf.get_scaled_positions()[None,:,:]+np.array(pbcs)[:,None,:],sf.cell))
        dists = cdist(relsurf.get_positions(), sfpbcpos)
        match = np.remainder(np.argmin(dists,axis=1),len(sf))
        if np.all(match == np.linspace(0,len(sf)-1,len(sf))) and np.max(np.min(dists,axis=1)) <0.01: # index is already alligned.
            initsurf = make_atoms_from_doc(datum['slab_information']['initial_configuration'])
            initsurf.set_tags(tags)
        else: # use pymatgen to align index
            sf2relsurf = mat.get_mapping(AseAtomsAdaptor.get_structure(sf),AseAtomsAdaptor.get_structure(relsurf))
            if sf2relsurf is not None:
                initsurf = make_atoms_from_doc(datum['slab_information']['initial_configuration'])
                initsurf = initsurf[sf2relsurf] # realign
                initsurf.set_tags(tags)
            else:
                continue # Cannot align slab+adsorbate calculations with the slab calculations
        ## Check if if number of neighbors are over 9. because CGCNN need it..
        structure = AseAtomsAdaptor.get_structure(initsurf)
        all_nbrs_init = structure.get_all_neighbors(radius,include_index=True)
        num_of_nbrs = [len(nbrs) for nbrs in all_nbrs_init]
        if np.min(num_of_nbrs) < nnbr:
            continue
        ## record data
        ### Process neighbors
        all_nbrs_init = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs_init]
        nbr_fea_idx_init = np.array([list(map(lambda x: x[2],nbr[:nnbr])) for nbr in all_nbrs_init]).tolist()
        nbr_fea_init = np.array([list(map(lambda x: x[1], nbr[:nnbr])) for nbr in all_nbrs_init]).tolist()
        ### Channel
        atoms_init = {'positions':initsurf.get_positions().tolist(),
            'cell':initsurf.get_cell().tolist(),'pbc':True,
            'numbers':initsurf.get_atomic_numbers().tolist(),'prop':datum['adsorption_energy'],
            'tags':tags.tolist(),'nbr_fea_idx':nbr_fea_idx_init,'nbr_fea':nbr_fea_init}
        datainitsurf.append(atoms_init)

    # Substitute
    for datum in datainitsurf:
        for i in np.where(np.array(datum['tags'])==1)[0]:
            datum['numbers'][i] = fakemap[datum['numbers'][i]]
    pickle.dump(datainitsurf,open(path,'wb'))
    
upperbound = 1.15

COdata = pickle.load(open('./CO_docs_slab_added.pkl','rb'))
ProcessData(COdata,6,upperbound,'./COdata.pickle')
Hdata = pickle.load(open('./H_docs_slab_added.pkl','rb'))
ProcessData(Hdata,1,upperbound,'./Hdata.pickle')


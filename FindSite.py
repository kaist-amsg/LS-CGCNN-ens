from scipy.spatial import Delaunay
import numpy as np
from collections import defaultdict
import copy
def alpha_shape_3D(pos, alpha,roundtol = 9):
    """
    Compute the alpha shape (concave hull) of a set of 3D points.
    Parameters:
        pos - np.array of shape (n,3) points.
        alpha - alpha value.
        roundtol - number of decimal for rounding 
    return
        outer surface vertex indices, edge indices, and triangle indices
    """

    tetra_vertices = Delaunay(pos).vertices
    # Find radius of the circumsphere.
    # By definition, radius of the sphere fitting inside the tetrahedral needs 
    # to be smaller than alpha value
    # http://mathworld.wolfram.com/Circumsphere.html
    tetrapos = np.take(pos,tetra_vertices,axis=0)
    normsq = np.sum(tetrapos**2,axis=2)[:,:,None]
    ones = np.ones((tetrapos.shape[0],tetrapos.shape[1],1))
    a = np.linalg.det(np.concatenate((tetrapos,ones),axis=2))
    Dx = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[1,2]],ones),axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,2]],ones),axis=2))
    Dz = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,1]],ones),axis=2))
    c = np.linalg.det(np.concatenate((normsq,tetrapos),axis=2))
    # Remove bad tetrahedrals. These the ones where volume is zero.
    bad = a==0
    num = Dx**2+Dy**2+Dz**2-4*a*c
    bad[num<0] = True
    bad = np.where(bad)[0]
    tetra_vertices = np.delete(tetra_vertices,bad,axis=0)
    num = np.delete(num,bad,axis=0)
    a = np.delete(a,bad,axis=0)
    # get radius
    r = np.sqrt(num)/(2*np.abs(a))

    # Find tetrahedrals
    tetras = tetra_vertices[r<alpha,:]
    # triangles
    TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    Triangles = tetras[:,TriComb].reshape(-1,3)
    Triangles = np.sort(Triangles,axis=1)
    # Remove triangles that occurs twice, because they are within shapes
    TrianglesDict = defaultdict(int)
    for tri in Triangles:TrianglesDict[tuple(tri)] += 1
    Triangles=np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])
    #edges
    if Triangles.size==0:
        return [], [], []
    EdgeComb=np.array([(0, 1), (0, 2), (1, 2)])
    Edges=Triangles[:,EdgeComb].reshape(-1,2)
    Edges=np.sort(Edges,axis=1)
    Edges=np.unique(Edges,axis=0)

    Vertices = np.unique(Edges)
    return Vertices,Edges,Triangles
    
pbcs = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [1, 1, 0], [-1, 1, 0], [0, -1, 0], [1, -1, 0], [-1, -1, 0], [0, 0, 1], [1, 0, 1], [-1, 0, 1], [0, 1, 1], [1, 1, 1], [-1, 1, 1], [0, -1, 1], [1, -1, 1], [-1, -1, 1], [0, 0, -1], [1, 0, -1], [-1, 0, -1], [0, 1, -1], [1, 1, -1], [-1, 1, -1], [0, -1, -1], [1, -1, -1], [-1, -1, -1]])
def FindSitesXYZ(cell,xyz,alpha=2.8): # Find site positions.
    xyz = np.array(xyz)
    cell = np.array(cell)
    xyz_frac = np.linalg.solve(cell.T, xyz.T).T
    xyz_frac = np.remainder(xyz_frac,1)
    # account for the pbc condition before applying alpha shape
    xyz_frac_pbcs = np.concatenate((pbcs.reshape(27,1,3)+xyz_frac.reshape(1,-1,3)),axis=0)
    xyz_pbcs = np.dot(xyz_frac_pbcs,cell)
    V,E,T = alpha_shape_3D(xyz_pbcs, alpha)
    if isinstance(V,list) or isinstance(V,np.ndarray) and V.size==0:
        return [],[],[],0.0
    # Find ones that are within the original cell
    V = V[V<len(xyz)]
    E = E[np.any(E<len(xyz),axis=1)]
    T = T[np.any(T<len(xyz),axis=1)]
    if V.size==0:
        return [],[],[],0.0
    # Remove duplicates
    E = E[np.unique(np.sort(np.remainder(E,len(xyz)),axis=1),axis=0,return_index=True)[1],:]
    T = T[np.unique(np.sort(np.remainder(T,len(xyz)),axis=1),axis=0,return_index=True)[1],:]
    
    TopXyz = xyz_pbcs[V,:]
    BridgeXyz = np.mean(xyz_pbcs[E,:],axis=1)
    HollowXyz = np.mean(xyz_pbcs[T,:],axis=1)
    
    temp1 = xyz_pbcs[T[:,[[0,1],[1,2],[0,2]]],:]
    triangle_lengths = np.linalg.norm(temp1[:,:,0,:] - temp1[:,:,1,:],axis=2)
    ph = np.sum(triangle_lengths,axis=1)/2
    area = np.sqrt(ph*(ph-triangle_lengths[:,0])*(ph-triangle_lengths[:,1])*(ph-triangle_lengths[:,2]))
    return TopXyz,BridgeXyz,HollowXyz, np.sum(area)

def FindSitesIdx(cell,xyz,alpha=2.8):
    xyz = np.array(xyz)
    cell = np.array(cell)
    xyz_frac = np.linalg.solve(cell.T, xyz.T).T
    xyz_frac = np.remainder(xyz_frac,1)
    # account for the pbc condition before applying alpha shape
    xyz_frac_pbcs = np.concatenate((pbcs.reshape(27,1,3)+xyz_frac.reshape(1,-1,3)),axis=0)
    xyz_pbcs = np.dot(xyz_frac_pbcs,cell)
    V,E,T = alpha_shape_3D(xyz_pbcs, alpha)
    if isinstance(V,list) or isinstance(V,np.ndarray) and V.size==0:
        return [],[],[],0.0
    # Find unique
    V = np.unique(np.remainder(V,len(xyz))).reshape(-1,1)
    E = np.unique(np.sort(np.remainder(E,len(xyz)),axis=1),axis=0)
    T = np.unique(np.sort(np.remainder(T,len(xyz)),axis=1),axis=0)
    if V.size==0:
        return [],[],[],0.0
    return V,E,T

if __name__ == '__main__':
    import json
    data_path = './example/example_data.json'
    label_map_path = './COparams/COInput_sub_map.json'
    data = json.load(open(data_path))
    label_map = json.load(open(label_map_path))
    NewData = []
    for d in data:
        V,E,T = FindSitesIdx(d['cell'],d['positions'])
        for site_idx in V.tolist()+E.tolist()+T.tolist():
            
            newdatum = copy.deepcopy(d)
            NoMap = False
            for i in site_idx:
                if str(newdatum['numbers'][i]) in label_map:
                    newdatum['numbers'][i] = label_map[str(newdatum['numbers'][i])]
                else:
                    NoMap=True
                    break
            if NoMap:
                continue
            newdatum['site_idx'] = site_idx
            NewData.append(newdatum)
    json.dump(NewData,open('./example/labeled_data.json','w'))
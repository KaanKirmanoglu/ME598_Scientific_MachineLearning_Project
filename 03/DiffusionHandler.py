import numpy as np
import gsd.hoomd
from hoomd.data import make_snapshot, boxdim
import sys
# from collections import defaultdict,deque
# import os.path
# import networkx as nx
# import itertools
# import time
# from sklearn.cluster import DBSCAN, OPTICS, MeanShift, estimate_bandwidth
# from scipy.spatial import cKDTree as KDTree
# import matplotlib.pyplot as plt
# import freud
# from scipy.spatial import distance
# from textwrap import wrap
# import string
# from scipy.spatial.transform import Rotation as Rot
np.set_printoptions(suppress=True,threshold=sys.maxsize)

"""
October 7 2022

First version
"""



def check_overlap2(pts1,pts2,margin,L):
    """
    Two groups of points returns False if they are closer than margin
    in the box
    """
    pts1 = pts1 + L*0.5
    pts2 = pts2 + L*0.5
    tree1 = KDTree(data=pts1, leafsize=12, boxsize=L+0.0001)
    tree2 = KDTree(data=pts2, leafsize=12, boxsize=L+0.0001)
    n = tree2.count_neighbors(tree1, margin)
    if(n==0):
        answer = True
    else:
        answer = False
    return answer

def translate(pos,v,L):
    pos = pos + v
    pos = pbc(pos,L)
    return pos

def rotatePbc(pos,rot_max,L):
    """
    Assumes the first is the central particle ie. CofM
    """
    cm = np.copy(pos[0])
    pos = pos - cm
    pos = np.matmul(rot_max,np.transpose(pos))
    pos = np.transpose(pos)
    pos = pos + cm
    pos = pbc(pos,L)
    return pos

def random_quat():
    """
    Not sure the first or last term is the scalar
    probably last
    """
    rands = np.random.uniform(size=3)
    quat = np.array([np.sqrt(1.0-rands[0])*np.sin(2*np.pi*rands[1]),
            np.sqrt(1.0-rands[0])*np.cos(2*np.pi*rands[1]),
            np.sqrt(rands[0])*np.sin(2*np.pi*rands[2]),
            np.sqrt(rands[0])*np.cos(2*np.pi*rands[2])])
    return quat

def quat_to_matrix(quat):
    """
    Convert a quaternion (assuming last term is scalar)
    to a rotation matrix that rotates columns when multiplies
    from left.
    https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Euler_rotations
    """
    i = quat[0]
    j = quat[1]
    k = quat[2]
    r = quat[3]
    matrix = np.zeros((3,3))
    matrix[0,0] = -1.0 + 2.0*i*i + 2.0*r*r
    matrix[0,1] = 2.0*(i*j-k*r)
    matrix[0,2] = 2.0*(i*k+j*r)
    matrix[1,0] = 2.0*(i*j+k*r)
    matrix[1,1] = -1.0 + 2.0*j*j + 2.0*r*r
    matrix[1,2] = 2.0*(j*k-i*r)
    matrix[2,0] = 2.0*(i*k-j*r)
    matrix[2,1] = 2.0*(j*k+i*r)
    matrix[2,2] = -1.0 + 2.0*k*k + 2.0*r*r
    return matrix

def quat_converter(quat):
    """
    HOOMD : scalar is first (ie default is 1,0,0,0)
    MCSim : scalar is last (ie default is 0,0,0,1)
    Converts mc_sim quats -> hoomd quats
    ovito also uses mc_sim quats
    """
    a,b,c,d = quat
    quat_new = np.array([d,a,b,c])
    return quat_new

def pbc(pos,L):
    pos = pos - ((pos - L*0.5)//L + 1)*L
    return pos

def rotate_rows(matrix,pos):
    """
    takes a position matrix (N,3)-> rows are positions
    and a rotation matrix to rotate the positions
    - simple r_mat *matmul* pos_mat rotates the columns only
    which is not what we want. It is possible to take
    transpose of row-wise position vector do the matmul
    and than retranspose it again to row wise to get the
    same result.
    """
    pos = np.matmul(matrix,np.transpose(pos))
    return np.transpose(pos)


def check_overlap(p,pts,margin,L):
    """
    Returns true if there is no overlap btw a point p and a  group of points pts
    Useful for placing noms
    """
    pts = np.array(pts)
    dists_v = pts - p
    dists_v = pbc(dists_v,L)
    dists = np.linalg.norm(dists_v,axis=1)
    min_dist = np.min(dists)
    if(min_dist>margin):
        answer = True
    else:
        answer = False
    return answer


def matrix_to_quat(m):
    """
    take a rotation matrix and return corresponding quat
    onenote 6.0
    !!! Different Quat Notation !!
    Unit quat is = [1,0,0,0]
    (before it was [0,0,0,1])
    This one unfortunately fails if the one of the euler angles are too close to
    zero(Not so sure about that) (quat returns NaN since inside the sqrt becomes
    negative) Scipy handles it better
    """
    q = np.zeros(4)
    q[0] = 0.5*np.sqrt(1.0+m[0,0]+m[1,1]+m[2,2])
    q[1] = (1.0/(4.0*q[0]))*(m[2,1]-m[1,2])
    q[2] = (1.0/(4.0*q[0]))*(m[0,2]-m[2,0])
    q[3] = (1.0/(4.0*q[0]))*(m[1,0]-m[0,1])
    return q

def matrix_to_quat2(m):
    """
    Try with scipy
    """
    r = Rot.from_dcm(m)
    q = r.as_quat()
    return q


def wrap_pbc(x, Box):
    delta = np.where(x > 0.5 * Box, x- Box, x)
    delta = np.where(delta <- 0.5 * Box, Box + delta, delta)
    return delta

def com(a,Box):
    theta = np.divide(a + 0.5 * Box, Box)*np.multiply(2,np.pi)
    xi_average = np.average(np.cos(theta), axis = 0)
    zeta_average = np.average(np.sin(theta), axis = 0)
    theta_average = np.arctan2(-zeta_average,-xi_average) + np.pi
    com = np.multiply(Box,theta_average)/np.multiply(2,np.pi)-0.5 * Box
    return com

def getMOI(pos):
    """
    Copied from : https://scipython.com/book/chapter-6-numpy/problems/p65/the-moment-of-inertia-tensor/
    Inertia products ie. non-diagonal terms can be negative
    """
    masses = np.ones(len(pos))
    x, y, z = pos.T
    Ixx = np.sum(masses * (y**2 + z**2))
    Iyy = np.sum(masses * (x**2 + z**2))
    Izz = np.sum(masses * (x**2 + y**2))
    Ixy = -np.sum(masses * x * y)
    Iyz = -np.sum(masses * y * z)
    Ixz = -np.sum(masses * x * z)
    I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    return I

class Shape():
    """
    Similar to MoleculeTemplate class in mcsim code.
    Any necessary info about the shape that is independent of the actual simulation box

    positions include the central particle at first index
    typeids are always 0 for central 1 for the constituents
    """
    def __init__(self,filename,dummy_type,balanced_shape):
        self.positions = []
        self.typeids = []
        self.filename = filename
        self.dummy_type = dummy_type
        self.balanced_shape = balanced_shape
        self.read_shape(filename,0)


    def set_positions(self):
        """
        Assumes Input shape is guaranteed to not cross any boundary
        Removes the filling and centers the shape to origin
        Adds the central particle
        """
        self.positions = self.positions[self.typeid!=self.dummy_type]
        self.typeid = self.typeid[self.typeid!=self.dummy_type]
        self.typeid[self.typeid>self.dummy_type] -= 1
        n_type = len(self.types) - 1
        alphabet_string = string.ascii_uppercase
        alphabet_list = list(alphabet_string)
        self.types = alphabet_list[:n_type]

        self.positions = self.positions - np.average(self.positions,axis=0)
        new_pos = np.zeros((len(self.positions)+1,3))
        new_pos[1:,:] = self.positions
        self.positions = new_pos
        self.typeid = np.ones(len(self.positions),dtype=int)
        self.typeid[0] = 0

        self.resetPosForInertia()

    def resetPosForInertia(self):
        """
        This class should store the shape positions with diagonal MOI
        """
        matrix_moi = getMOI(self.positions[1:])
        w,v = np.linalg.eigh(matrix_moi)
        self.diagonalMOI = w
        Q = v
        R = np.linalg.inv(Q)
        perfect_pos = rotate_rows(R,self.positions)
        if(self.balanced_shape==0):
            self.positions = perfect_pos
        self.mass = np.ones(len(self.positions),dtype=float)
        self.mass[0] = len(self.positions)
        self.MOI = np.zeros_like(self.positions)
        self.MOI[0,:] = self.diagonalMOI
        self.body = np.zeros_like(self.typeid,dtype=int)
        a = np.array([1.0,0.0,0.0,0.0])
        a = np.tile(a,(len(self.positions),1))
        self.orientation = a




    def getPositions(self):
        return self.positions
    def getTypeids(self):
        return self.typeid
    def getmass(self):
        return self.mass
    def getbody(self):
        return self.body
    def getMOI(self):
        return self.MOI
    def getorientation(self):
        return self.orientation

    def read_shape(self,input,target_frame):
        try:
            with gsd.hoomd.open(name=input, mode='rb') as f:
                if (target_frame==-1):
                    frame = f.read_frame(len(f)-1)
                    self.frame = len(f)-1
                    print("Reading last frame ")
                else:
                    self.frame = target_frame
                    frame = f.read_frame(target_frame)
                    print("Reading frame ", target_frame)
                self.positions = (frame.particles.position).copy()
                self.types = (frame.particles.types).copy()
                self.typeid = (frame.particles.typeid).copy()

        except:
            self.positions = (input.particles.position).copy()
            self.types = (input.particles.types).copy()
            self.typeid = (input.particles.typeid).copy()

def check_overlap_solvent(p,pts,margin,L):
    pts = np.array(pts)
    dists_v = pts - p
    dists_v = pbc(dists_v,L)
    dists = np.linalg.norm(dists_v,axis=1)
    min_dist = np.min(dists)
    if(min_dist>margin):
        answer = True
    else:
        answer = False
    return answer


class DiffusionHandler():
    """
    used in diffusion.hoomd
    """
    def __init__(self,L,rho,d1):

        self.Lx = L
        self.rho = rho
        self.rA = d1*0.5
        self.sigmaAB = self.rA + 0.5
        self.cutAB = 2.0**(1/6.)*self.sigmaAB

        self.N = np.int(np.power(L,3.0)*rho)

    def placeBeads(self):
        """
        October 7 - diffusion.hoomd
        Randomly place the solvent beads so that they won't overlap with tracer
        that is assumed to be at the center
        """
        tracer_extra_margin = 0.5
        solvent_margin = 1.1

        first_pos = [
        [self.Lx*0.4,self.Lx*0.4,self.Lx*0.4],
        [-self.Lx*0.4,-self.Lx*0.4,-self.Lx*0.4],
        [self.Lx*0.4,-self.Lx*0.4,-self.Lx*0.4]
        ]

        first_types = [1,1,1]

        pos = first_pos
        typeids = first_types

        for i in range(self.N - len(first_types)):
            good1 = False
            good2 = False
            while(good1 is False or good2 is False):
                p = (np.random.rand(3) - 0.5)*self.Lx
                good1 = check_overlap_solvent(p,pos,solvent_margin,self.Lx)
                if(np.linalg.norm(p)>(tracer_extra_margin+self.rA)):
                    good2 = True
                else:
                    good2 = False
            pos.append(p)
            typeids.append(1)

        # add the tracer
        pos.append([0.0,0.0,0.0])
        typeids.append(0)

        self.typeid = typeids
        self.positions = pos
        self.types = ['Eu','Gd']
        # self.velocities = np.zeros_like(self.positions)

    def getSigmaAB(self):
        """
        sigma for LJ
        """
        return self.sigmaAB

    def getCutAB(self):
        """
        get the cutoff for to make LJ->WCA for A-B interaction
        see script wca_plotter.py
        """
        return self.cutAB

    def get_snap(self,context):
        with context:


            snap = make_snapshot(N=len(self.positions),
                                particle_types=self.types,
                                box=boxdim(Lx=self.Lx,Ly=self.Lx,Lz=self.Lx))


            for k in range(len(self.positions)):
                snap.particles.position[k] = self.positions[k]
                snap.particles.typeid[k] = self.typeid[k]
                # snap.particles.body[k] = self.body[k]
                # snap.particles.moment_inertia[k] = self.moment_inertia[k]
                # snap.particles.orientation[k] = self.orientations[k]
                # snap.particles.mass[k] = self.mass[k]
                # snap.particles.velocity[k] = self.velocities[k]
                # snap.particles.angmom[k] = self.angmom[k]

                # snap.particles.charge[k] = self.charges[k]
            # set bond typeids and groups
            # snap.bonds.resize(len(self.bond_group))
            # for k in range(len(self.bond_group)):
            #     snap.bonds.typeid[k] = self.bond_typeid[k]
            #     snap.bonds.group[k] = self.bond_group[k]
            # if(len(snap.particles.body)>4):
            #     snap.particles.body[snap.particles.body[:] > 5000]=-1

            # print(self.body)
            # print(type(snap.particles.body))
            # exit()

        return snap

##################### ASAGISINI SALLLLLLLLLLLLLLLLLLL #######################

    def read_system(self,input,target_frame):
        """
        Read in a snapshot from a gsd file or snapshot.
        """
        try:
            with gsd.hoomd.open(name=input, mode='rb') as f:
                if (target_frame==-1):
                    frame = f.read_frame(len(f)-1)
                    self.frame = len(f)-1
                    print("Reading last frame ")
                else:
                    self.frame = target_frame
                    frame = f.read_frame(target_frame)
                    print("Reading frame ", target_frame)
                self.positions = (frame.particles.position).copy()
                self.velocities = (frame.particles.velocity).copy()
                self.moleculeid = (frame.particles.charge).copy()
                self.charges = (frame.particles.charge).copy()

                self.types = (frame.particles.types).copy()
                self.typeid = (frame.particles.typeid).copy()

                self.bond_types = (frame.bonds.types).copy()
                self.bond_group = (frame.bonds.group).copy()
                self.bond_typeid = (frame.bonds.typeid).copy()

                self.Lx,self.Ly,self.Lz = frame.configuration.box[0:3]
                self.box = frame.configuration.box

        except:
            self.positions = (input.particles.position).copy()
            self.velocities = (input.particles.velocity).copy()
            self.types = (input.particles.types).copy()
            self.typeid = (input.particles.typeid).copy()
            self.moleculeid = (input.particles.charge).copy()
            self.charges = (input.particles.charge).copy()

            self.bond_types = (input.bonds.types).copy()
            self.bond_group = (input.bonds.group).copy()
            self.bond_typeid = (input.bonds.typeid).copy()

            self.Lx = input.box.Lx
            self.Ly = input.box.Lx
            self.Lz = input.box.Lx
            self.box = input.box

    def setShape(self,dummy_type,balanced_shape):
        """
        Create the shape class
        """
        self.shape = Shape(self.shape_filename,dummy_type,balanced_shape)
        self.shape.set_positions()

    def setBox(self,Box_size,N_nom):
        """
        set the nom concentration box sixe etc
        """
        self.Lx = Box_size
        self.Ly = Box_size
        self.Lz = Box_size
        self.NomPerShape = N_nom
        self.shape_concentration = 1.0/(Box_size**3)
        self.Nshapes = 1
        self.placeInitialParticles()

    def placeInitialParticles(self):
        """
        Place the one shape and the noms such that no two beads are closer than
        margin
        margin for shape -> since the shape is empty not to risk to put inside
        the shape shape beads have a larger margin for NOM beads, this value
        could be given as a parameter

        Put the shape as it is in the template class so that it will have unit orientation
        """
        margin = 1.5
        shape_margin = 3.0
        pos = self.shape.getPositions()
        typeids = self.shape.getTypeids()
        # quat_mcsim = random_quat()
        # rot_max = quat_to_matrix(quat_mcsim)
        # pos = rotate_rows(rot_max,pos)
        # quat_hoomd = quat_converter(quat_mcsim)

        ### This 3 lines are very bad put 2 noms at the corners beforehand
        ### so that checkoverlap works
        new_noms = [[0.5*self.Lx-0.5,0.5*self.Lx-0.5,0.5*self.Lx-0.5],[-0.5*self.Lx+0.5,-0.5*self.Lx+0.5,-0.5*self.Lx+0.5]]
        new_nom_typeid = [2,2]
        for i in range(self.NomPerShape-2):
            good1 = False
            good2 = False
            while(good1 is False or good2 is False):
                p = (np.random.rand(3) - 0.5)*self.Lx
                good1 = check_overlap(p,pos,shape_margin,self.Lx)
                good2 = check_overlap(p,new_noms,margin,self.Lx)
            new_noms.append(p)
            new_nom_typeid.append(2)
        pos = np.vstack((pos,new_noms))
        typeids = np.hstack((typeids,new_nom_typeid))
        # a = np.array([1.0,0.0,0.0,0.0])
        # a = np.tile(a,(len(pos),1))
        # a[0,:] = quat_hoomd
        # self.orientations = a
        self.typeid = typeids
        self.positions = pos
        self.types = ['R','A','B']
        self.velocities = np.zeros_like(self.positions)
        self.angmom = np.zeros((len(self.positions),4))

    def setRigidBody(self):
        """
        Sets rigid body related things like mass, body, inertia etc.
        """
        mass_shape = self.shape.getmass()
        body_shape = self.shape.getbody()
        inertia_shape = self.shape.getMOI()
        orientation_shape = self.shape.getorientation()
        mass_noms = np.ones(self.NomPerShape)
        body_noms = np.ones(self.NomPerShape)*-1.0

        inertia_noms = np.zeros((self.NomPerShape,3))
        orientation_noms = np.tile([1.0,0.0,0.0,0.0],(self.NomPerShape,1))
        self.mass = np.hstack((mass_shape,mass_noms))
        self.body = np.hstack((body_shape,body_noms))
        self.moment_inertia = np.vstack((inertia_shape,inertia_noms))
        self.orientations = np.vstack((orientation_shape,orientation_noms))

        l1 = len(self.positions)
        l2 = len(self.typeid)
        l3 = len(self.body)
        l4 = len(self.moment_inertia)
        l5 = len(self.orientations)
        l6 = len(self.mass)
        print(l1,l2,l3,l4,l5,l6)



    def getRigidConstituents(self):
        """
        Return the shape template positions except the central
        """
        pos = self.shape.getPositions()
        pos = pos[1:]
        return pos

    def readSnapshot(self,snapshot):
        input = snapshot
        try:
            print("Reading in : " + input)
        except:
            print("Reading a snapshot not gsd file")

        try:
            with gsd.hoomd.open(name=input, mode='rb') as f:
                frame = f.read_frame(len(f)-1)
                self.positions = (frame.particles.position).copy()
                self.velocities = (frame.particles.velocity).copy()
                self.types = (frame.particles.types).copy()
                self.typeid = (frame.particles.typeid).copy()
                self.Lx,self.Ly,self.Lz = frame.configuration.box[0:3]
                print("Should've read from snapshot not file!!! ")
                exit()

        except:
            self.positions = (input.particles.position).copy()
            self.velocities = (input.particles.velocity).copy()
            self.types = (input.particles.types).copy()
            self.typeid = (input.particles.typeid).copy()
            self.bodi = (input.particles.body).copy()
            self.moment_inertia = (input.particles.moment_inertia).copy()
            self.orientations = (input.particles.orientation).copy()
            self.mass = (input.particles.mass).copy()
            self.angmom = (input.particles.angmom).copy()

            self.Lx = input.box.Lx
            self.Ly = input.box.Lx
            self.Lz = input.box.Lx

        """
        snapshot.particles.body is broken, non body particles should have -1
        but the container I think only holds unsigned ints so -1 defaults to
        4294967295 which is very inconvenient for the rest oof the class
        so here I swithc it back to -1
        """
        self.body = np.array(self.bodi,dtype=int)
        self.body[self.body>9999999]= -1.0

    def addShape(self):
        """
        add the new shape far away from the other shapes
        than distribute the noms that it overlaps with
        Central particles must have a body tag identical to their contiguous tag.
        """
        t = 0
        current_body = len(self.positions)
        pos_new_shape = self.shape.getPositions()
        typeid_new_shape = self.shape.getTypeids()
        velocity_new_shape = np.zeros_like(pos_new_shape)
        angmom_new_shape = np.zeros((len(pos_new_shape),4))
        pos_old_shapes = self.positions[self.body>-0.5]
        good = False
        margin = 3.0
        while(good is False):
            pos_new_shape = self.shape.getPositions()
            t += 1
            p = (np.random.rand(3) - 0.5)*self.Lx*0.85 ### 0.70 to make encounter earlier see 28.1
            quat = random_quat()
            rot_max = quat_to_matrix(quat)
            quat_hoomd = quat_converter(quat)
            pos_new_shape = translate(pos_new_shape,p,self.Lx)
            pos_new_shape = rotatePbc(pos_new_shape,rot_max,self.Lx)
            good = check_overlap2(pos_new_shape,pos_old_shapes,margin,self.Lx)
            if(t>100000):
                print("AddShape tried too much!!!")
                exit()
        self.positions = np.vstack((self.positions,pos_new_shape))
        self.typeid = np.hstack((self.typeid,typeid_new_shape))
        self.velocities = np.vstack((self.velocities,velocity_new_shape))
        self.angmom = np.vstack((self.angmom,angmom_new_shape))
        # print("quat", quat_hoomd)

        mass_shape = self.shape.getmass()
        body_shape = np.ones_like(mass_shape)*current_body
        inertia_shape = self.shape.getMOI()
        orientation_shape = self.shape.getorientation()
        # print(orientation_shape[:10])
        orientation_shape[0,:] = quat_hoomd
        # print(orientation_shape[:10])
        self.mass = np.hstack((self.mass,mass_shape))
        self.body = np.hstack((self.body,body_shape))
        self.moment_inertia = np.vstack((self.moment_inertia,inertia_shape))
        self.orientations = np.vstack((self.orientations,orientation_shape))
        self.redistributeNoms(pos_new_shape)

        self.Nshapes += 1

    def redistributeNoms(self,pos_new_shape):
        """
        After adding the shape must collocate the overlapping noms
        Margin must be large enough to make sure that nothing stays inside the
        molecule
        """
        margin = 3.0
        ## Since I know it doesn't overlap with molecules anything it overlaps
        ## is an NOM
        for i,p in enumerate(self.positions):
            if(self.typeid[i]==2):
                good = check_overlap(p,pos_new_shape,margin,self.Lx)
                while(good is False):
                    p = (np.random.rand(3) - 0.5)*self.Lx
                    ## in the new random positions check overlap with all other shapes
                    good = check_overlap(p,self.positions,margin,self.Lx)
                self.positions[i] = p

    def addNoms(self):
        """
        Add [self.NomPerShape] Noms
        """
        margin_nom = 2.0
        shape_margin = 3.0
        new_nom_pos = []
        for i in range(1):
            good = False
            while(good is False):
                p = (np.random.rand(3) - 0.5)*self.Lx
                good = check_overlap(p,self.positions,shape_margin,self.Lx)
            new_nom_pos.append(p)

        for i in range(1):
            good1 = False
            good2 = False
            while(good1 is False or good2 is False):
                p = (np.random.rand(3) - 0.5)*self.Lx
                good1 = check_overlap(p,self.positions,shape_margin,self.Lx)
                if(np.linalg.norm(pbc((p[0]-p[1]) ,self.Lx  ))>2.0):
                    good2 = True
            new_nom_pos.append(p)

        for i in range(self.NomPerShape-2):
            good1 = False
            good2 = False
            while(good1 is False or good2 is False):
                p = (np.random.rand(3) - 0.5)*self.Lx
                good1 = check_overlap(p,self.positions,shape_margin,self.Lx)
                good2 = check_overlap(p,new_nom_pos,margin_nom,self.Lx)
            new_nom_pos.append(p)

        new_nom_typeid = np.ones(self.NomPerShape,dtype=int)*2
        new_nom_mass = np.ones(self.NomPerShape)
        new_nom_body = np.ones(self.NomPerShape,dtype=int)*-1
        new_inertia_noms = np.zeros((self.NomPerShape,3))
        new_orientation_noms = np.tile([1.0,0.0,0.0,0.0],(self.NomPerShape,1))
        new_nom_velocities = np.zeros_like(new_nom_pos)
        new_nom_angmoms = np.zeros_like(new_orientation_noms)

        self.positions = np.vstack((self.positions,new_nom_pos))
        self.typeid = np.hstack((self.typeid,new_nom_typeid))
        self.velocities = np.vstack((self.velocities,new_nom_velocities))
        self.angmom = np.vstack((self.angmom,new_nom_angmoms))
        self.mass = np.hstack((self.mass,new_nom_mass))
        self.body = np.hstack((self.body,new_nom_body))
        self.moment_inertia = np.vstack((self.moment_inertia,new_inertia_noms))
        self.orientations = np.vstack((self.orientations,new_orientation_noms))

        l = []
        l.append(len(self.positions))
        l.append(len(self.typeid))
        l.append(len(self.velocities))
        l.append(len(self.angmom))
        l.append(len(self.mass))
        l.append(len(self.body))
        l.append(len(self.moment_inertia))
        l.append(len(self.orientations))

        l = np.array(l)
        u = np.unique(l)
        if(len(u)>1.5):
            print("Warning 34-RK")
            print(l)

    def isOneCluster(self,cutoff,min_samples):
        self.clusterize(cutoff,min_samples)
        cluster_of_shapes = self.cluster_info[self.typeid==1]
        unq = np.unique(cluster_of_shapes)
        answer = True
        if(unq[0]!=0):
            print("Something wrong with cluster check snapshot !!")
            exit()
        if(len(unq)!=1):
            answer = False
        return answer

    def adjustTheBoxSize(self,cutoff,min_samples):
        """
        Mostly copied from mc_sim CanonicalSim::enlargeBox()
        Tricky part is to keep cluster intact while you enlarge the box

        1st Hopeful Approach
        Just center the cluster to the box and then hope that it doesn't cross
        any boundaries. Than just increase the box size

        It is actually better to do this before adding the shape than I don't
        have to worry about the last shape when I enlarge the box

        1 - Make sure there is only one cluster
        2 - Center the cluster
        3 - Make sure it doesnt cross boundary (with before boxsize)
        4 - Increase box size

        """
        L_before = np.copy(self.Lx)
        L_after = np.power((self.Nshapes + 1)/self.shape_concentration, 1.0/3.0)
        print(" The box size change : %.2f -> %.2f" %(L_before,L_after))
        self.centerCluster2()
        good,disp = self.checkClusterAfter()
        tt = 0
        while(good is False):
            tt += 1
            if(tt>1000):
                print("Cluster can't fit to the box !!!!")
                exit()
            self.shiftCluster(disp)
            good,disp = self.checkClusterAfter()
        self.Lx = L_after
        self.Ly = L_after
        self.Lz = L_after

    def clusterize(self,cutoff,min_samples):
        """
        For now pick the largest cluster put it to the center of the box
        Resize the box if necessary
        Keep the Noms
        """
        isNoNoise = False
        self.cutoff = cutoff
        self.min_samples = min_samples
        self.posplus = self.positions + self.Lx*0.5
        tree = KDTree(data=self.posplus, leafsize=12, boxsize=self.Lx+0.00001)
        pairs = tree.sparse_distance_matrix(tree,cutoff+1.0)
        if(cutoff<1.0):
            print("cutoff too small shapes will be divided !!!")
            exit()
        dbscan = DBSCAN(eps=cutoff, min_samples=min_samples, metric="precomputed", n_jobs=-1)
        labels0 = dbscan.fit_predict(pairs)
        n_clusters0 = len(set(labels0)) - (1 if -1 in labels0 else 0)
        if -1 not in labels0 :
            isNoNoise=True
        n0,cluster_count = np.unique(labels0,return_counts=True)
        cluster_info = np.ones_like(self.positions)
        cluster_info = np.multiply(cluster_info,999)
        cluster_info[:,0] = labels0
        if(isNoNoise):
            cluster_info[:,1] = cluster_count[labels0]
        else:
            cluster_info[:,1] = cluster_count[labels0+1]
            n0 = np.delete(n0,0)
        self.cluster_info = cluster_info[:,0]
        self.cluster_ids = n0

    def centerCluster2(self):
        """
        1 - Find the COM of the cluster
        2 - Instead of translating cluster and redistributing the noms just
        tranlate everything in the box
        """

        pos_cluster = self.positions[self.cluster_info==0]
        typeid_cluster = self.typeid[self.cluster_info==0]
        N_nom_to_add = len(self.typeid[self.typeid==1]) - len(typeid_cluster[typeid_cluster==1])
        cluster_cm = com(pos_cluster,self.Lx)
        new_pos = self.positions - cluster_cm
        new_pos = wrap_pbc(new_pos,self.Lx)
        self.positions = new_pos

    def CheckClusterBefore(self):
        """
        This one is not used for now. I think it is probably unnecessary because
        instead of translating the cluster to center I simply translate everthing
        in the box such that origin becomes the cOfM of the cluster so I believe
        checkClusterAfter() should be enough.

        Only concern is whether the com() function is able to calculate the cofmass
        for clusters larger than half the box size or not
        -> see the drawing at the later parts of OneNote 28.1
        """
        pos_cluster = self.positions[self.cluster_info==0]
        cm_cluster = com(pos_cluster,self.Lx)


    def checkClusterAfter(self):
        """
        Checks if the extremes of the cluster are too close to boundaries or not
        Also check that distance from the cm of the cluster to the extremes of the
        cluster is less than half the box size This makes sure that cluster can be
        translated noicely
        """
        pos_cluster = self.positions[self.cluster_info==0]
        x = np.array([np.min(pos_cluster[:,0]),np.max(pos_cluster[:,0])])
        y = np.array([np.min(pos_cluster[:,1]),np.max(pos_cluster[:,1])])
        z = np.array([np.min(pos_cluster[:,2]),np.max(pos_cluster[:,2])])
        walls = np.array([self.Lx*(-0.5), self.Lx*(0.5)])
        dists = []
        dists.append(x[0]-walls[0])
        dists.append(walls[1]-x[1])
        dists.append(y[0]-walls[0])
        dists.append(walls[1]-y[1])
        dists.append(z[0]-walls[0])
        dists.append(walls[1]-z[1])
        min_dist = np.min(dists)

        good = True
        disp = False
        disps = np.array([
        [1.0,0.0,0.0],
        [-1.0,0.0,0.0],
        [0.0,1.0,0.0],
        [0.0,-1.0,0.0],
        [0.0,0.0,1.0],
        [0.0,0.0,-1.0],
        ])

        if(min_dist<0.7):
            direction = np.argmin(dists)
            disp = disps[direction]
            print("Cluster will be shifted by :")
            print(disp)
            good = False

        return good,disp

    def shiftCluster(self,disp):
        """
        Move everything a little so that cluster doesn't hit the boundaries
        """

        new_pos = self.positions + disp*np.random.rand()*3.0
        new_pos = wrap_pbc(new_pos,self.Lx)
        self.positions = new_pos


    def return_snapshot(self):
        snap = make_snapshot(N=len(self.positions),
                            particle_types=self.types,
                            # bond_types=self.bond_types,
                            # angle_types=self.angle_types,
                            box=boxdim(Lx=self.Lx,Ly=self.Ly,Lz=self.Lz))



        for k in range(len(self.positions)):
            snap.particles.position[k] = self.positions[k]
            snap.particles.typeid[k] = self.typeid[k]
            # snap.particles.charge[k] = self.charges[k]
            snap.particles.body[k] = self.body[k]
            snap.particles.moment_inertia[k] = self.moment_inertia[k]
            snap.particles.orientation[k] = self.orientations[k]
            snap.particles.mass[k] = self.mass[k]
            snap.particles.velocity[k] = self.velocities[k]
            snap.particles.angmom[k] = self.angmom[k]
        # set bond typeids and groups
        # snap.bonds.resize(len(self.bond_group))
        # for k in range(len(self.bond_group)):
        #     snap.bonds.typeid[k] = self.bond_typeid[k]
        #     snap.bonds.group[k] = self.bond_group[k]
        if(len(snap.particles.body)>4):
            snap.particles.body[snap.particles.body[:] > 5000]=-1





        return snap

    def setContinueSim(self,input,target_frame):
        """
        Set :
        self.NomPerShape
        self.shape_concentration
        self.Nshapes
        Basically do the task of setBox() for continued simulations
        """
        with gsd.hoomd.open(name=input, mode='rb') as f:
            if (target_frame==-1):
                frame = f.read_frame(len(f)-1)
                print("Reading last frame ")
            else:
                frame = f.read_frame(target_frame)
                print("Reading frame ", target_frame)
            self.positions = (frame.particles.position).copy()
            self.velocities = (frame.particles.velocity).copy()
            self.body = (frame.particles.body).copy()
            self.moment_inertia = (frame.particles.moment_inertia).copy()
            self.orientations = (frame.particles.orientation).copy()
            self.mass = (frame.particles.mass).copy()
            self.typeid = (frame.particles.typeid).copy()

            self.types = (frame.particles.types).copy()

            self.Lx,self.Ly,self.Lz = frame.configuration.box[0:3]

        self.Nshapes = len(self.positions[self.typeid==0])
        N_nom = len(self.positions[self.typeid==2])
        self.NomPerShape = int(N_nom/self.Nshapes)
        self.shape_concentration = self.Nshapes/(self.Lx**3)
        print(" $$$$$ $$$$$ $$$$ $$$$ $$$$ $$$$ $$$$$ $$$$$  $$$$ $ $$$$$ ")
        print(" - - - - - - - - - - - - - - - - - -- - -- - ---- ---- - ---- - - - ")
        print("Continue Simulation with : ")
        print("# of shapes : %d" %(self.Nshapes))
        print("# of NOMs per shape  : %d" %(self.NomPerShape))
        print(" - - - - - - - - - - - - - - - - - -- - -- - ---- ---- - ---- - - - ")
        print(" $$$$$ $$$$$ $$$$ $$$$ $$$$ $$$$ $$$$$ $$$$$  $$$$ $ $$$$$ ")
        return self.Nshapes


########################################## ASAGIYI SAL ############################################

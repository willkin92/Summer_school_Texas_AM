import numpy as np

# Isotropic tensor L for given K and mu
def isotropic(K,mu,L):
    L=np.matrix([[K+4*mu/3, K-2*mu/3, K-2*mu/3,\
                  0., 0., 0.],\
                 [K-2*mu/3, K+4*mu/3, K-2*mu/3,\
                  0., 0., 0.],\
                 [K-2*mu/3, K-2*mu/3, K+4*mu/3,\
                  0., 0., 0.],\
                 [0., 0., 0., mu, 0., 0.],\
                 [0., 0., 0., 0., mu, 0.],\
                 [0., 0., 0., 0., 0., mu]])
    return L
    

# Orthotropic tensor L for given engineering constants
def orthotropic(Vprop,L):
    Ex=Vprop[0]
    Ey=Vprop[1]
    Ez=Vprop[2]
    vxy=Vprop[3]
    vxz=Vprop[4]
    vyz=Vprop[5]
    muxy=Vprop[6]
    muxz=Vprop[7]
    muyz=Vprop[8]
    M=np.matrix([[1/Ex, -vxy/Ex, -vxz/Ex, 0., 0., 0.],\
                 [-vxy/Ex, 1/Ey, -vyz/Ey, 0., 0., 0.],\
                 [-vxz/Ex, -vyz/Ey, 1/Ez, 0., 0., 0.],\
                 [0., 0., 0., 1/muxy, 0., 0.],\
                 [0., 0., 0., 0., 1/muxz, 0.],\
                 [0., 0., 0., 0., 0., 1/muyz]])
    L=M.I             
    return L

    
# Engineering constants for given orthotropic tensor L
def orthoinv(L,Vprop):
    M=L.I
    Vprop[0]=1/M[0,0]
    Vprop[1]=1/M[1,1]
    Vprop[2]=1/M[2,2]
    Vprop[3]=-M[0,1]*Vprop[0]
    Vprop[4]=-M[0,2]*Vprop[0]
    Vprop[5]=-M[1,2]*Vprop[1]
    Vprop[6]=1/M[3,3]
    Vprop[7]=1/M[4,4]
    Vprop[8]=1/M[5,5]
    return Vprop
    

import numpy as np
from elastic import isotropic, orthotropic, orthoinv
from polar_sc import polarization

# Data
E0=3.5
v0=0.3
Vprop1=np.array([14.5, 14.5, 230.0, 0.5104, 0.0162, 0.0162,\
                 4.8, 22.8, 22.8])
c=4.13832/31.7592

# Elasticity tensors
K0=E0/(3*(1-2*v0))
mu0=E0/(2*(1+v0))
L0=np.zeros((6,6))
L1=np.zeros((6,6))
L0=isotropic(K0,mu0,L0)
L1=orthotropic(Vprop1,L1)

#Polarization tensor
P1=np.zeros((6,6))
P1=polarization(K0,mu0,1.,"ec",P1)

# Interaction tensor
I=np.eye(6)
T1=(I+P1*(L1-L0)).I

# Mori-Tanaka strain concentration tensor of fiber and matrix
A1=T1*((1-c)*I+c*T1).I
A0=(I-c*A1)/(1-c)

# Effective elastic properties
Leff=(1-c)*L0*A0+c*L1*A1              
Vprop1=orthoinv(Leff,Vprop1)
print(Vprop1)

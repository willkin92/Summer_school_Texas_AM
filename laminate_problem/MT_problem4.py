import numpy as np
from elastic import isotropic
from polar_sc import polarization
# Data
E0=3
v0=0.3
E1=10
v1=0.2
c1=0.2
I=np.eye(6)

# Elasticity tensors
K0=E0/(3*(1-2*v0))
mu0=E0/(2*(1+v0))
K1=E1/(3*(1-2*v1))
mu1=E1/(2*(1+v1))
L0=np.zeros((6,6))
L1=np.zeros((6,6))
L0=isotropic(K0,mu0,L0)
L1=isotropic(K1,mu1,L1)

# polarization tensor
P1=np.zeros((6,6))
P1=polarization(K0,mu0,1.,"ec",P1)

# Interaction tensor
T1=(I+P1*(L1-L0)).I
# Mori-Tanaka strain concentration tensors
A1=T1*((1-c1)*I+c1*T1).I
A0=(I-c1*A1)/(1-c1)
# Effective elastic properties
Lmt=(1-c1)*L0*A0+c1*L1*A1
np.savetxt('Lmt.txt',Lmt,fmt='%1.2f')

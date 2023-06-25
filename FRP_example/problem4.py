import numpy as np
from elastic import isotropic, orthotropic, orthoinv
from polar_sc import polarization
from matplotlib import pyplot as plt
# Data
E0=3
v0=0.3
E=np.linspace(10,100,10)
Ez=[]
Ex=[]
for E1 in E:
    v1=0.2
    Vprop1=np.array([E1, E1, E1, v1, v1, v1,\
                    E1/(2*(1+v1)), E1/(2*(1+v1)), E1/(2*(1+v1))])
   #what should be there
    c=0.2
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
    orthoinv(Leff,Vprop1)
    print(Vprop1)
    Ez.append(Vprop1[2])
    Ex.append(Vprop1[0])
plt.figure(1)
plt.plot(E,Ez)
plt.xlabel('E')
plt.ylabel('Ez')
plt.grid()
plt.show()
plt.figure(2)
plt.plot(E,Ex)
plt.xlabel('E')
plt.ylabel('Ex')
plt.grid()
plt.show()

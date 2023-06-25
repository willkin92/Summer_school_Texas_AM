import numpy as np
from elastic import isotropic,orthoinv
from MT_problem4 import Lmt
from matplotlib import pyplot as plt
# Data
E=np.linspace(1,5,10)
Ez=[]
Ex=[]
for E2 in E:
    v2=0.25
    c=0.3
    # Elasticity tensors
    L1=Lmt
    K2=E2/(3*(1-2*v2))
    mu2=E2/(2*(1+v2))
    L2=np.zeros((6,6))
    L2=isotropic(K2,mu2,L2)

    # mn and mt
    Lnn1=np.matrix([[L1[0,0], L1[0,3], L1[0,4]],\
                    [L1[0,3], L1[3,3], L1[3,4]],\
                    [L1[0,4], L1[3,4], L1[4,4]]])
    Lnt1=np.matrix([[L1[0,1], L1[0,2], L1[0,5]],\
                    [L1[1,3], L1[2,3], L1[3,5]],\
                    [L1[1,4], L1[2,4], L1[4,5]]])
    Lnn2=np.matrix([[L2[0,0], L2[0,3], L2[0,4]],\
                    [L2[0,3], L2[3,3], L2[3,4]],\
                    [L2[0,4], L2[3,4], L2[4,4]]])
    Lnt2=np.matrix([[L2[0,1], L2[0,2], L2[0,5]],\
                    [L2[1,3], L2[2,3], L2[3,5]],\
                    [L2[1,4], L2[2,4], L2[4,5]]])
    mn=((1-c)*Lnn1.I+c*Lnn2.I).I
    mt=mn*((1-c)*Lnn1.I*Lnt1+c*Lnn2.I*Lnt2)

    # Strain concentration tensors
    dUndx1=Lnn1.I*(mn-Lnn1)
    dUtdx1=Lnn1.I*(mt-Lnt1)
    dUndx2=Lnn2.I*(mn-Lnn2)
    dUtdx2=Lnn2.I*(mt-Lnt2)
    A1=np.matrix([[1.+dUndx1[0,0], dUtdx1[0,0], dUtdx1[0,1],\
                dUndx1[0,1], dUndx1[0,2], dUtdx1[0,2]],\
                [0., 1., 0., 0., 0., 0.],\
                [0., 0., 1., 0., 0., 0.],\
                [dUndx1[1,0], dUtdx1[1,0], dUtdx1[1,1],\
                1.+dUndx1[1,1], dUndx1[1,2], dUtdx1[1,2]],\
                [dUndx1[2,0], dUtdx1[2,0], dUtdx1[2,1],\
                dUndx1[2,1], 1.+dUndx1[2,2], dUtdx1[2,2]],\
                [0., 0., 0., 0., 0., 1.]])
    A2=np.matrix([[1.+dUndx2[0,0], dUtdx2[0,0], dUtdx2[0,1],\
                dUndx2[0,1], dUndx2[0,2], dUtdx2[0,2]],\
                [0., 1., 0., 0., 0., 0.],\
                [0., 0., 1., 0., 0., 0.],\
                [dUndx2[1,0], dUtdx2[1,0], dUtdx2[1,1],\
                1.+dUndx2[1,1], dUndx2[1,2], dUtdx2[1,2]],\
                [dUndx2[2,0], dUtdx2[2,0], dUtdx2[2,1],\
                dUndx2[2,1], 1.+dUndx2[2,2], dUtdx2[2,2]],\
                [0., 0., 0., 0., 0., 1.]])

    # Effective properties
    Lper=(1-c)*L1*A1+c*L2*A2
    np.savetxt('Lper.txt',Lper,fmt='%1.2f')
    Vprop=np.zeros(9)
    Vprop=orthoinv(Lper,Vprop)
    print('Ex=',Vprop[0])
    print('Ey=',Vprop[1])
    print('Ez=',Vprop[2])
    print('vxy=',Vprop[3])
    print('vxz=',Vprop[4])
    print('vyz=',Vprop[5])
    print('muxy=',Vprop[6])
    print('muxz=',Vprop[7])
    print('muyz=',Vprop[8])
    Ez.append(Vprop[2])
    Ex.append(Vprop[0])
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

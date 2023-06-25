import numpy as np

######################
# Mechanical problem # 
######################

def polarization(K0,mu0,r,typ,P):
# all cases refer to isotropic matrix

# prolate spheroid: a1=a2, r=a3/a1
    if typ=="ps":
       Z=(3*K0+mu0)/(4*mu0*(3*K0+4*mu0))
       ea=(r*r-1)*(r*r-1)*(r*r-1)
       I1=(r/np.sqrt(ea))*(r*np.sqrt(r*r-1)-np.arccosh(r))
       w=1/(r*r-1)
       P11=((5*mu0-3*K0)/(3*K0+mu0)-9*w/4)*I1+3*(1+w)/2
       P12=-(1+3*w/4)*I1+(1+w)/2
       P13=(2+3*w)*I1-2*(1+w)
       P33=-2*((6*K0+8*mu0)/(3*K0+mu0)+3*w)*I1+2*((6*K0+8*mu0)/(3*K0+mu0)+2*w)
       P44=3*(4*mu0/(3*K0+mu0)-w)*I1+2*(1+w)
       P55=6*(3*K0/(3*K0+mu0)+2*w)*I1-4*((3*K0-2*mu0)/(3*K0+mu0)+2*w)
       P=Z*np.matrix([[P11, P12, P13, 0., 0., 0.],\
                      [P12, P11, P13, 0., 0., 0.],\
                      [P13, P13, P33, 0., 0., 0.],\
                      [0., 0., 0., P44, 0., 0.],\
                      [0., 0., 0., 0., P55, 0.],\
                      [0., 0., 0., 0., 0., P55]])

# oblate spheroid: a1=a2, r=a3/a1
    if typ=="os":
       Z=(3*K0+mu0)/(4*mu0*(3*K0+4*mu0))
       ea=(1-r*r)*(1-r*r)*(1-r*r)
       I1=(r/np.sqrt(ea))*(np.arccos(r)-r*np.sqrt(1-r*r))
       w=1/(r*r-1)
       P11=((5*mu0-3*K0)/(3*K0+mu0)-9*w/4)*I1+3*(1+w)/2
       P12=-(1+3*w/4)*I1+(1+w)/2
       P13=(2+3*w)*I1-2*(1+w)
       P33=-2*((6*K0+8*mu0)/(3*K0+mu0)+3*w)*I1+2*((6*K0+8*mu0)/(3*K0+mu0)+2*w)
       P44=3*(4*mu0/(3*K0+mu0)-w)*I1+2*(1+w)
       P55=6*(3*K0/(3*K0+mu0)+2*w)*I1-4*((3*K0-2*mu0)/(3*K0+mu0)+2*w)
       P=Z*np.matrix([[P11, P12, P13, 0., 0., 0.],\
                      [P12, P11, P13, 0., 0., 0.],\
                      [P13, P13, P33, 0., 0., 0.],\
                      [0., 0., 0., P44, 0., 0.],\
                      [0., 0., 0., 0., P55, 0.],\
                      [0., 0., 0., 0., 0., P55]])

# elliptic cylinder: r=a1/a2, a3-->infinity
    if typ=="ec":
       Z=(1/((1+r)*(1+r)))*((3*K0+mu0)/(2*mu0*(3*K0+4*mu0)))
       P11=(6*mu0/(3*K0+mu0))*(1+r)+r
       P12=-r
       P22=r*(1-r+P11)
       P44=P11+P22+2*(r*r-r+1)
       P55=P11+2+r
       P66=r*P55
       P=Z*np.matrix([[P11, P12, 0., 0., 0., 0.],\
                      [P12, P22, 0., 0., 0., 0.],\
                      [0., 0., 0., 0., 0., 0.],\
                      [0., 0., 0., P44, 0., 0.],\
                      [0., 0., 0., 0., P55, 0.],\
                      [0., 0., 0., 0., 0., P66]])

# spherical inclusion: a1=a2=a3, r=1
    if typ=="si":
       PK=(1/3)*(1/(3*K0+4*mu0))
       Pmu=(1/5)*(1/(3*K0+4*mu0)+1/(2*mu0))
       P=np.matrix([[PK+4*Pmu/3, PK-2*Pmu/3, PK-2*Pmu/3, 0., 0., 0.],\
                    [PK-2*Pmu/3, PK+4*Pmu/3, PK-2*Pmu/3, 0., 0., 0.],\
                    [PK-2*Pmu/3, PK-2*Pmu/3, PK+4*Pmu/3, 0., 0., 0.],\
                    [0., 0., 0., 4*Pmu, 0., 0.],\
                    [0., 0., 0., 0., 4*Pmu, 0.],\
                    [0., 0., 0., 0., 0., 4*Pmu]])

# disk like: a2=a3, a1=0, r=1
    if typ=="dl":
       P=np.matrix([[3/(3*K0+4*mu0), 0., 0., 0., 0., 0.],\
                    [0., 0., 0., 0., 0., 0.],\
                    [0., 0., 0., 0., 0., 0.],\
                    [0., 0., 0., 1/mu0, 0., 0.],\
                    [0., 0., 0., 0., 1/mu0, 0.],\
                    [0., 0., 0., 0., 0., 0]])

    return P


def polar_cyl(Ktr0,mutr0,muax0,P):
# long cylinder inclusion and transversely isotropic matrix
    P=np.matrix([[3/(Ktr0+mutr0)+1/mutr0, 1/(Ktr0+mutr0)-1/mutr0, 0., 0., 0., 0.],\
                 [1/(Ktr0+mutr0)-1/mutr0, 3/(Ktr0+mutr0)+1/mutr0, 0., 0., 0., 0.],\
                 [0., 0., 0., 0., 0., 0.],\
                 [0., 0., 0., 4/(Ktr0+mutr0)+4/mutr0, 0., 0.],\
                 [0., 0., 0., 0., 4/muax0, 0.],\
                 [0., 0., 0., 0., 0., 4/muax0]])
    return P



def polar_lamin(L11,L14,L15,L44,L45,L55,P):
# disk-like inclusion and anisotropic matrix
    Ln=np.matrix([[L11, L14, L15], [L14, L44, L45], [L15, L45, L55]])
    Pcal=Ln.I
    P=np.matrix([[Pcal[0,0], 0., 0., Pcal[0,1], Pcal[0,2], 0.],\
                 [0., 0., 0., 0., 0., 0.],\
                 [0., 0., 0., 0., 0., 0.],\
                 [Pcal[0,1], 0., 0., Pcal[1,1], Pcal[1,2], 0.],\
                 [Pcal[0,2], 0., 0., Pcal[1,2], Pcal[2,2], 0.],\
                 [0., 0., 0., 0., 0., 0.]])
    return P




###################
# Thermal problem # 
###################


def polarizK(k0,r,typ,Pk):
# all cases refer to isotropic matrix

# prolate spheroid: a1=a2, r=a3/a1
    if typ=="ps":
       ea=(r*r-1)*(r*r-1)*(r*r-1)
       I1=(r/np.sqrt(ea))*(r*np.sqrt(r*r-1)-np.arccosh(r))
       P11=I1/(2*k0)
       P33=1/k0-2*P11
       Pk=np.matrix([[P11, 0., 0.],\
                     [0., P11, 0.],\
                     [0., 0., P33]])

# oblate spheroid: a1=a2, r=a3/a1
    if typ=="os":
       Z=(3*K0+mu0)/(4*mu0*(3*K0+4*mu0))
       ea=(1-r*r)*(1-r*r)*(1-r*r)
       I1=(r/np.sqrt(ea))*(np.arccos(r)-r*np.sqrt(1-r*r))
       P11=I1/(2*k0)
       P33=1/k0-2*P11
       Pk=np.matrix([[P11, 0., 0.],\
                     [0., P11, 0.],\
                     [0., 0., P33]])

# elliptic cylinder: r=a1/a2, a3-->infinity
    if typ=="ec":
       P11=1/(k0*(1+r))
       P22=r/(k0*(1+r))
       Pk=np.matrix([[P11, 0., 0.],\
                     [0., P22, 0.],\
                     [0., 0., 0.]])

# spherical inclusion: a1=a2=a3, r=1
    if typ=="si":
       Pk=np.matrix([[1/(3*k0), 0., 0.],\
                     [0., 1/(3*k0), 0.],\
                     [0., 0., 1/(3*k0)]])

# disk like: a2=a3, a1=0, r=1
    if typ=="dl":
       Pk=np.matrix([[1/k0, 0., 0.],\
                     [0., 0., 0.],\
                     [0., 0., 0.]])

    return Pk




def polarizK_cyl(ktr0,Pk):
# long cylinder inclusion and transversely isotropic matrix
    Pk=np.matrix([[1/(2*ktr0), 0., 0.],\
                  [0., 1/(2*ktr0), 0.],\
                  [0., 0., 0.]])
    return Pk


def polarizK_lamin(k11,Pk):
# disk-like inclusion and anisotropic matrix
    Pk=np.matrix([[1/k11, 0., 0.],\
                  [0., 0., 0.],\
                  [0., 0., 0.]])
    return Pk

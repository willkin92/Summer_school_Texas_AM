import numpy as np
from MT_cylinder import Lmt, L0 as Lm, L1 as Lf, A0 as Am, A1 as Af, c1 as cf 
from PH_layered import Lper, L2, A2, A1 as Amt, c as c2
# Data
Ms=np.matrix([[50.], [0.], [0.], [0.], [0.], [0.]])
# Macroscale
Me=Lper.I*Ms
# Mesoscale
e2=A2*Me
s2=L2*e2
emt=Amt*Me
# Microscale
ef=Af*emt
sf=Lf*ef
em=Am*emt
sm=Lm*em
# Write results
f = open('fields.txt', 'w')
f.write("e2 s2 ef sf em sm\n")
for i in range(6):
    f.write("%1.5f,%1.2f,%1.5f,%1.2f,%1.5f,%1.2f\n" %\
           (e2[[i]],s2[[i]],ef[[i]],sf[[i]],em[[i]],sm[[i]]))
f.close()

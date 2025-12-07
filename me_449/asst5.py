import modern_robotics as mr
import numpy as np
import pdb

# 8.2
r =.02
h=.2
rho=7500

m=rho*h*np.pi*.02**2
mcyl = m
print("Mcyl", mcyl)

ixx=m/12*(3*r**2+h**2)
izz=m/5*(2*r**2)
ixx1 = ixx
iyy1 = ixx
izz1 = izz
print("Icyl", ixx1, iyy1, izz1)

m = rho*4/3*np.pi*.1**3
msph=m
dsph=.1+.1
ixx=m/5*2*.1**2
ixx2=iyy2=izz2=ixx
print("Isph", ixx2, iyy2, izz2)
print("msph", msph)

# parallel axis
ixxa = ixx1+2*ixx2
iyya = iyy1+2*iyy2
izza = izz1+2*(izz2+msph*dsph**2)
print("Itot", ixxa, iyya, izza)

# Gb
Ib = np.zeros([3,3])
Ib[0,0] = ixxa
Ib[1,1] = iyya
Ib[2,2] = izza

mass = mcyl + 2*msph
massI = mass * np.identity(3)

zeros = np.zeros([3,3])

Gb = np.block([[Ib, zeros],
               [zeros, massI]])
print(Gb)





# 11.6
print("\n#11.6")
m=4
b=2
k=0.1

wn = np.sqrt(k/m)
damp = b/(2*np.sqrt(k*m))

print(wn, damp)

sb = wn*np.sqrt(damp**2-1)

s1 = -damp*wn+sb
s2 = -damp*wn-sb
print(s1, s2)

t1 = -1/s1
t2 = -1/s2

print(t1, t2)
twopercentss = 4 *t1
print(twopercentss)

bprime = 2*np.sqrt(k*m)
print(bprime)

kp = 611523.9
xinf = kp / (k+kp)
einf = 1 - xinf
finf = einf * kp
print(xinf, einf, finf)
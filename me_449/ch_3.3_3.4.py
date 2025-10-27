import modern_robotics as mr
import numpy as np

# https://github.com/NxRLab/ModernRobotics

# #8)
Rsa = np.array([[0, 1, 0],
              [0, 0, 1],
              [1, 0, 0]])

#1
xa = np.array([[0,0,1]]).T
ya = np.array([[-1,0,0]]).T
za = np.cross(xa.T, ya.T).T

xb = np.array([[1,0,0]]).T
yb = np.array([[0,0,-1]]).T
zb = np.cross(xb.T, yb.T).T

oa = np.array([[0,0,1]]).T
ob = np.array([[0,2,0]]).T

Ra = np.hstack((xa, ya, za))
Rb = np.hstack((xb, yb, zb))

Tsa = mr.RpToTrans(Ra, oa)
print("Tsa\n", Tsa)

#2
Tsb = mr.RpToTrans(Rb, ob)
print("Tsb\n", Tsb)
print("Tsb^-1\n", mr.TransInv(Tsb))

#3
Tab = mr.TransInv(Tsa) @ Tsb
print("Tab\n", Tab)

#4
T = Tsb
print("s frame")

#5
pb = np.array([[1,2,3,1]]).T
ps = Tsb @ pb
print("ps", ps)

#6
ps = np.array([[1,2,3,1]]).T
p = ps
print("no")

#7
vs = np.array([[3,2,1,-1,-2,-3]]).T
adjoint = mr.Adjoint(mr.TransInv(Tsa))
va = adjoint @ vs
print("va", va.T)

#8
log_Tsa = mr.MatrixLog6(Tsa)
print("log Tsa", log_Tsa)

#9
stheta = np.array([0,1,2,3,0,0])
exp_stheta = mr.MatrixExp6(mr.VecTose3(stheta))
print("exp stheta", exp_stheta)

#10
Fb = np.array([[1,0,0,2,1,0]]).T
adj_Tbs = mr.Adjoint(Tsb.T) # the transpose is included
Fs = adj_Tbs @ Fb
print("Fs", Fs.T)

#11
T = np.array([[0, -1, 0, 3],
              [1, 0, 0, 0],
              [0, 0, 1, 1],
              [0, 0, 0, 1]])
T_inv = mr.TransInv(T)
print("T-inv", T_inv)

#12
v = np.array([[1,0,0,0,2,3]]).T
se3 = mr.VecTose3(v.squeeze())
print("se3", se3)

#13
s_hat = np.array([[1,0,0]]).T
p = np.array([[0,0,2]]).T
h=1
axis = mr.ScrewToAxis(p.squeeze(), s_hat.squeeze(), h)
print("axis", axis)

#14
stheta = np.array([[0, -1.5708, 0, 2.3562],
                   [1.5708, 0, 0, -2.3562],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0]])
T=mr.MatrixExp6(stheta)
print("T", T)

#15
T=np.array([[0, -1, 0, 3],
            [1,0,0,0],
            [0,0,1,1],
            [0,0,0,1]])
se3 = mr.MatrixLog6(T)
print("se3", se3)
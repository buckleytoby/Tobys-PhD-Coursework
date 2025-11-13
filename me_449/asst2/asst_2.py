import modern_robotics as mr
import numpy as np

PI = np.pi


# Assignment 2, due 1:30 PM CT Wednesday October 15. Exercises 3.16, 3.27 (also draw the frame), 4.2, 4.5, 4.10 (also give the space and body Jacobians when the robot is at its home configuration), and 5.2. 

# 3.16 part i)
Tsa = np.array([[0, -1, 0, 3],
                [0, 0, -1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1]])

se3 = mr.MatrixLog6(Tsa)
print("se3", se3)

# my math
a = np.array([[3, -1, -1,],
              [1,3,-1,],
              [1,1,3]])
b = np.array([[2,4,2],
              [-2,2,4],
              [-4,-2,2]])

g = 3/2/PI*a / 3 + 1/6/np.sqrt(3)*b
print(g)
p=np.array([3,0,0]).T

v=g @ p
print("v", v)

theta = 2*PI/3
w = np.array([1,-1,1]).T /np.sqrt(3)
so3_w = mr.VecToso3(w)
print("[w]theta", so3_w *theta)
print("vtheta", v * theta)

# actual answer
vec = mr.se3ToVec(se3) / theta
print("S", vec)

h = np.array([.57, -.57, .57]).T.T @ np.array([2.2, -2.2, -1.4]) / 2.1
print("h", h)

s_hat = np.array([.57, -.57, .57])
theta_dot = theta
sq = -(v - h * s_hat * theta_dot) / theta_dot
print("sq", sq)
skew_s_hat = mr.VecToso3(s_hat)

# (a, b) where a @ x = b ... and we have skew_s_hat @ q = sq
q = np.linalg.lstsq(skew_s_hat, sq)[0]



#j)
st = np.array([0,1,2,3,0,0])
sst = mr.VecTose3(st)
T = mr.MatrixExp6(sst)
print("T", T)


# 4.2
l1 = l2 = l0 = 1
M = np.array([[1, 0, 0, 0],
              [0, 1, 0, l1+l2],
              [0, 0, 1, l0],
              [0, 0, 0, 1]])
Js = np.array([[0,0,0,0],
               [0,0,0,0],
               [1,1,1,0],
               [0, l1, l1+l2,0],
               [0,0,0,0],
               [0,0,0,1]])
Jb = np.array([[0,0,0,0],
               [0,0,0,0],
               [1,1,1,0],
               [-(l1+l2), -l2, 0, 0],
               [0,0,0,0],
               [0,0,0,1]])

th = np.array([0, PI/2, -PI/2, 1])

fks = mr.FKinSpace(M, Js, th)
fkb = mr.FKinBody(M, Jb, th)

print("fks", fks)
print("fkb", fkb)

pass
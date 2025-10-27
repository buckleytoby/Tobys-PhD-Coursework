import modern_robotics as mr
import math
import numpy as np


"""
The 6R UR5 robot is shown below at its home configuration. Eight frames are defined: the fixed frame {s} at the base, frames {1} through {6} attached to links 1 through 6, and the end-effector frame {b} which is fixed relative to link 6. (The frame {b} is not shown in the image.) The red arrow is the x-axis, the green arrow is the y-axis, and the blue arrow is the z-axis. Frames {s} and {1}-{6} are aligned when the robot is at its home configuration, i.e., each rotation matrix R i j {\displaystyle R_{ij}} (where i , j {\displaystyle i,j} could be s {\displaystyle s} or any number 1 through 6) is the identity matrix. 


The rotation axes for joint i {\displaystyle i}, defined in frame { i {\displaystyle i}}, are ω ^ 1 = ( 0 , 0 , 1 ) , ω ^ 5 = ( 0 , 0 , − 1 ) , ω ^ 2 = ω ^ 3 = ω ^ 4 = ω ^ 6 = ( 0 , 1 , 0 ) {\displaystyle {\hat {\omega }}_{1}=(0,0,1),{\hat {\omega }}_{5}=(0,0,-1),{\hat {\omega }}_{2}={\hat {\omega }}_{3}={\hat {\omega }}_{4}={\hat {\omega }}_{6}=(0,1,0)}.

For some set of joint angles θ {\displaystyle \theta }, we have the following relations between the orientations of the joint frames:

    R13 = [[-0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, -0.7071]]
    Rs2 = [[-0.6964, 0.1736, 0.6964], [-0.1228, -0.9848, 0.1228], [0.7071, 0, 0.7071]]
    R25 = [[-0.7566, -0.1198, -0.6428], [-0.1564, 0.9877, 0], [0.6348, 0.1005, -0.7661]]
    R12 = [[0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, 0.7071]]
    R34 = [[0.6428, 0, -0.7660], [0, 1, 0], [0.7660, 0, 0.6428]]
    Rs6 = [[0.9418, 0.3249, -0.0859], [0.3249, -0.9456, -0.0151], [-0.0861, -0.0136, -0.9962]]
    R6b = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
"""

"""
Find the six-vector of joint angles θ {\displaystyle \theta } given the R i j {\displaystyle R_{ij}} above. (You will likely want to calculate the rotation matrices R s , 1 {\displaystyle R_{s,1}} and R i , i + 1 {\displaystyle R_{i,i+1}} and use the MR code library, e.g., MatrixLog3.)
"""
w1 =np.array( [0,0,1])
w2 =np.array( [0, 1, 0])
w3 =np.array( w2)
w4 =np.array( w2)
w5 =np.array( [0, 0, -1])
w6 =np.array( w2)

R13 = np.array([[-0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, -0.7071]])
Rs2 = np.array([[-0.6964, 0.1736, 0.6964], [-0.1228, -0.9848, 0.1228], [0.7071, 0, 0.7071]])
R25 = np.array([[-0.7566, -0.1198, -0.6428], [-0.1564, 0.9877, 0], [0.6348, 0.1005, -0.7661]])
R12 = np.array([[0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, 0.7071]])
R34 = np.array([[0.6428, 0, -0.7660], [0, 1, 0], [0.7660, 0, 0.6428]])
Rs6 = np.array([[0.9418, 0.3249, -0.0859], [0.3249, -0.9456, -0.0151], [-0.0861, -0.0136, -0.9962]])
R6b = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

so3_1 = mr.VecToso3(w1)
so3_2 = mr.VecToso3(w2)
so3_3 = mr.VecToso3(w3)
so3_4 = mr.VecToso3(w4)
so3_5 = mr.VecToso3(w5)
so3_6 = mr.VecToso3(w6)

log12 = mr.MatrixLog3(R12)
v12 = mr.so3ToVec(log12)
a2_test = v12 @ w2.T

Rs1 = Rs2 @ R12.T
R23 = R12.T @ R13
R45 =  R34.T @ R23.T @ R25
R56 = (Rs1 @ R12 @ R23 @ R34 @ R45).T @ Rs6

a1 = mr.so3ToVec(mr.MatrixLog3(Rs1)) @ w1.T
a2 = mr.so3ToVec(mr.MatrixLog3(R12)) @ w2.T
a3 = mr.so3ToVec(mr.MatrixLog3(R23)) @ w3.T
a4 = mr.so3ToVec(mr.MatrixLog3(R34)) @ w4.T
a5 = mr.so3ToVec(mr.MatrixLog3(R45)) @ w5.T
a6 = mr.so3ToVec(mr.MatrixLog3(R56)) @ w6.T

angles = [a1, a2, a3, a4, a5, a6]

for i in range(len(angles)):
    print("angle {}: {}".format(i+1, angles[i]))

Rsb = Rs1 @ R12 @ R23 @ R34 @ R45 @ R56 @ R6b
print("Rsb", Rsb)

pass
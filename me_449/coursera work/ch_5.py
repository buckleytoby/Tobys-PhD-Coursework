import modern_robotics as mr
import numpy as np

"""
  S1=001000, S2=100020, S3=000010. 
"""
S = np.array([[0,0,1,0,0,0],
              [1,0,0,0,2,0],
              [0,0,0,0,1,0]]).T

th = np.array([np.pi / 2.0, np.pi/ 2.0, 1.0])

Js = mr.JacobianSpace(S, th)

print("part 3", Js)

"""
B1=010300, B2=−100030, B3=000001.
"""
B = np.array([[0,1,0,3,0,0],
              [-1,0,0,0,3,0],
              [0,0,0,0,0,1]]).T

Jb = mr.JacobianBody(B, th)
print("part 4: \n", Jb)

"""
"""

Jbv = np.array([[-.105, 0, 0.006, -.045, 0, .006, 0],
                [-0.889, .006, 0, -.844, .006, 0, 0],
                [0, -.105, .889, 0, 0, 0, 0]])

A = Jbv @ Jbv.T

eval, evecs = np.linalg.eig(A)

i = np.argmax(np.abs(eval))

max_evec = evecs[i]

v1 = max_evec
v2 = v1 / np.linalg.norm(v1)
print("part 5\n", v2)

print("part 6\n", np.sqrt(eval[i]))
import modern_robotics as mr
import numpy as np

pi = np.pi
#4
# θ=(−π/2,π/2,π/3,−π/4,1,π/6)
theta = np.array((-pi/2, pi/2, pi/3, -pi/4, 1, pi/6))

M = np.array([[1,0,0,3.73],
                [0,1,0,0],
                [0,0,1,2.73],
                [0,0,0,1]])

Slist = np.array([[0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [1, 0, 0, 0, 0, 1],
                    [0, 0, 1, -0.73, 0, 0],
                    [-1, 0, 0, 0, 0, -3.73],
                    [0, 1, 2.73, 3.73, 1, 0]])

T = mr.FKinSpace(M, Slist, theta)

print("#4", T)

#4
# θ=(−π/2,π/2,π/3,−π/4,1,π/6) (same as above)
Blist = np.array([[0, 0, 0, 0, 0, 0],
[0, 1, 1, 1, 0, 0],
[1, 0, 0, 0, 0, 1],
[0, 2.73, 3.73, 2, 0, 0],
[2.73, 0, 0, 0, 0, 0],
[0, -2.73, -1, 0, 1, 0]])

T = mr.FKinBody(M, Blist, theta)
print("#5", T)
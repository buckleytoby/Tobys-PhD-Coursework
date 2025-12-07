import modern_robotics as mr
import numpy as np

#2
th0 = np.array([np.pi/4, np.pi/4, np.pi/4]).T
Tsd = np.array([[-.585, -.811, 0, .076],
                [.811, -.585, 0, 2.608],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

epsw = 0.001
epsv = 0.0001

M = np.array([[1, 0, 0, 3],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# screw axes (space)
Slist = np.array([[0, 0, 1, 0, 0, 0],
                  [0, 0, 1, 0, -1, 0],
                  [0, 0, 1, 0, -2, 0]]).T

th = mr.IKinSpace(Slist, M, Tsd, th0, epsw, epsv)
print(th)
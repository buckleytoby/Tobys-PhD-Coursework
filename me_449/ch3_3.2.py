import modern_robotics as mr
import numpy as np

# https://github.com/NxRLab/ModernRobotics

# #8)
Rsa = np.array([[0, 1, 0],
              [0, 0, 1],
              [1, 0, 0]])

log_Rsa = mr.MatrixLog3(Rsa)
w3 = -1.0 * log_Rsa[0][1]
w2 = log_Rsa[0][2]
w1 = -1.0 * log_Rsa[1][2]

w = np.array([w1, w2, w3])
w_norm = np.linalg.norm(w)
print(w_norm)

# #9)
omega_theta = np.array([1,2,0])
norm = np.linalg.norm(omega_theta)
omega_hat = omega_theta / norm
theta = norm

w1, w2, w3 = omega_theta
skew_omega_theta = np.array([[0, -w3, w2],
                             [w3, 0, -w1],
                             [-w2, w1, 0]])


print(mr.MatrixExp3(skew_omega_theta))

#10
print(mr.VecToso3(np.array([1,2,0.5])))

# 11
m = np.array([[0, 0.5, -1],
              [-0.5, 0, 2],
              [1, -2, 0]])

print(mr.MatrixExp3(m))

# 12
R = np.array([[0, 0, 1],
              [-1, 0, 0],
              [0, -1, 0]])

print(mr.MatrixLog3(R))
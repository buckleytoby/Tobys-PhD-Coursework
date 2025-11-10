import modern_robotics as mr
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class Tf(np.ndarray):
    """
    a config, aka a transform. 4x4
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # must be 4x4
        assert(self.shape == [4, 4])

class Twist(np.ndarray):
    """
    a twist, 6x1, [w; v]
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # must be 6x1
        assert(self.shape == [6, 1])

def fk(M, Blist, q):
    """
    returns the FK transform of q
    """
    Tsb = mr.FKinBody(M, Blist, q)
    # Tsb = Tf(Tsb)
    return Tsb

def error_twist(Tsb, Tsd) -> Twist:
    """
    computes the twist to get from T1 to T2 in 1 second. Assumes that T1 and T2 are w.r.t. the same frame
    """
    Tsb_inv = mr.TransInv(Tsb)
    Tbd = Tsb_inv @ Tsd

    # convert from 4x4 transform matrix to a 4x4 log(T) mat
    log_Tbd = mr.MatrixLog6(Tbd)

    # go from the log 4x4 to the twist 6x1
    v = mr.se3ToVec(log_Tbd)

    v = v.reshape([6, 1])
    return v

# part 1
"""
Each iteration reports the iteration number i, the joint
vector θi, the end-effector configuration Tsb(θi), the error twist Vb, and the angular and linear error
magnitudes, ∥ωb∥ and ∥vb∥
"""
def IKinBodyIterates(xd, q0, M, Blist):    
    # params
    nb_joints = q0.shape[0]

    # loop vars
    c = 0
    eps_r = 0.001
    eps_l = 0.0001
    er_mag = 99.0
    el_mag = 99.0

    # history
    q_history = []
    history = defaultdict(list)

    # joints
    if q0 is None:
        q = np.zeros([nb_joints, 1])
    else:
        q = q0

    assert(q.shape == (nb_joints, 1))

    # ee config
    Tsb = fk(M, Blist, q)
    error_t = error_twist(Tsb, xd)

    while er_mag > eps_r or el_mag > eps_l:
        # increment the counter
        c += 1

        # early exit
        if c > 100:
            print("Didn't converge.")
            break

        # compute the jacobian
        J = mr.JacobianBody(Blist, q)

        # invert the jacobian using np
        J_inv = np.linalg.pinv(J)

        # compute the delta-x
        # dx = xd - Tsb

        # compute the delta-q
        dq = J_inv @ error_t

        # compute the new q
        q = q + dq

        # wrap the joints
        q = np.atan2(np.sin(q), np.cos(q))

        # compute new Tsb
        Tsb = fk(M, Blist, q)

        # compute the error twist
        error_t = error_twist(Tsb, xd)

        # compute the ang error
        er = error_t[0:3]

        # compute the linear error
        el = error_t[3:6]

        # compute the error mags
        er_mag = np.linalg.norm(er)
        el_mag = np.linalg.norm(el)

        # save as a row
        history['q'].append(q.T)
        history['c'].append(c)
        history['xyz'].append(Tsb[0:3, 3].T)
        history['el_mag'].append(el_mag)
        history['er_mag'].append(er_mag)

        # printing
        print("Iteration: {}".format(c))
        print("Joint Vector: {}".format(q))
        print("Tsb: {}".format(Tsb))
        print("Error twist: {}".format(error_t))
        print("Error rot mag: {}".format(er_mag))
        print("Error lin mag: {}".format(el_mag))
        pass

    return history


########################################
# Part 1 UR5 Test
xd = np.array([[0, 0, -1, 0],
                    [1, 0, 0, 0.6],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1]])

# xd = Tf(xd)

# q0 = np.array([[0, 0, 0, 0, 0, 0]]).T
q0 = np.random.random([6, 1]) * np.pi * 2.0 - np.pi # [-pi, pi]

# computing the B matrix
# values taken from example 4.5 in MR
# W1 = 109 mm, W2 = 82 mm, L1 = 425 mm, L2 = 392 mm, H1 = 89 mm, H2 = 95 mm.
L1 = 0.425
L2 = 0.392
W1 = 0.109
W2 = 0.082
H1 = 0.089
H2 = 0.095
M = np.array([[-1, 0, 0, L1+L2],
              [0, 0, 1, W1+W2],
              [0, 1, 0, H1-H2],
              [0, 0, 0, 1]])

# M −1[Si]M
# Bi = np.invert(M) @ mr.VecTose3(Si) @ M

"""
joint 1: (0, 1, 0, W1 + W2, 0, L1 + L2)
joint 2: (0, 0, 1, H2, −L1 − L2, 0)
joint 3: (0, 0, 1, H2, −L2, 0)
joint 4: (0, 0, 1, H2, 0, 0)
joint 5: (0, −1, 0, −W2, 0, 0)
joint 6: (0, 0, 1, 0, 0, 0)
"""

Blist_t = np.array([[0, 1, 0, W1+W2, 0, L1+L2],
                    [0, 0, 1, H2, -L1-L2, 0],
                    [0, 0, 1, H2, -L2, 0],
                    [0, 0, 1, H2, 0, 0],
                    [0, -1, 0, -W2, 0, 0],
                    [0, 0, 1, 0, 0, 0]])
Blist = Blist_t.T

history1 = IKinBodyIterates(xd, q0, M, Blist)

# slightly perturb the answer
q = history1['q'][-1].copy()
q *= 1.05
q0 = q.reshape([6, 1])

history2 = IKinBodyIterates(xd, q0, M, Blist)

# plotting
f1 = plt.figure()
f1.add_subplot(111, projection='3d')
f2 = plt.figure()
f3 = plt.figure()

for history in [history1, history2]:
    # stack q_history as rows
    qh = np.array(history['q'])
    qh = qh.squeeze()

    # plotting
    """
    1. A 3D plot that shows the (x, y, z) position of the end-effector at each iterate, with a line
    between successive iterations. (See Figure 1 for an example.)
    2. A plot of the magnitude of the linear error (on the y-axis) as a function of the iterate number
    (on the x-axis).
    3. A plot of the magnitude of the angular error as a function of the iterate numbe
    """
    # 1
    xyz = np.array(history['xyz'])
    plt.figure(f1.number)
    ax = plt.gca()
    
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    ax.set_xlabel("EE X")
    ax.set_ylabel("EE Y")
    ax.set_zlabel("EE Z")

    # 2
    plt.figure(f2.number)
    plt.plot(history['c'], history['el_mag'])
    plt.title("Linear Error Mag.")
    plt.xlabel("Iteration Count")
    plt.ylabel("Linear Error Mag")

    # 3
    plt.figure(f3.number)
    plt.plot(history['c'], history['er_mag'])
    plt.title("Angular Error Mag.")
    plt.xlabel("Iteration Count")
    plt.ylabel("Angular Error Mag")


    # saving
    np.savetxt("./asst3_nb_iter_{}.csv".format(history['c'][-1]), qh, delimiter=',')
    pass

f1.legend(["Long Iterate", "Short Iterate"])
f1.show()
f1.savefig("./xyz.png")

f2.legend(["Long Iterate", "Short Iterate"])
f2.show()
f2.savefig("./linear_err.png")

f3.legend(["Long Iterate", "Short Iterate"])
f3.show()
f3.savefig("./ang_err.png")
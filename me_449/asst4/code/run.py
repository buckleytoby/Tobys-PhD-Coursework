import modern_robotics as mr
import numpy as np
from collections import defaultdict
import tqdm


# computing the B matrix
# values taken from example 4.5 in MR
# W1 = 109 mm, W2 = 82 mm, L1 = 425 mm, L2 = 392 mm, H1 = 89 mm, H2 = 95 mm.
L1 = 0.425
L2 = 0.392
W1 = 0.109
W2 = 0.082
H1 = 0.089
H2 = 0.095
UR5_M = np.array([[-1, 0, 0, L1+L2],
              [0, 0, 1, W1+W2],
              [0, 1, 0, H1-H2],
              [0, 0, 0, 1]])

class Tf():
    """
    a config, aka a transform. 4x4
    """
    def __init__(self, d):
        self.d = d

        # must be 4x4
        assert(self.d.shape == (4, 4))

    def getorigin(self):
        o = self.d[0:3, 3:4]

        assert(o.shape == (3, 1))
        return o
    
    def getrot(self):
        r = self.d[0:3, 0:3]

        assert(r.shape == (3, 3))
        return r
    
    def adj(self):
        # MR page 111, or def 3.20
        R = self.getrot()

        p = self.getorigin()

        # [p]R
        pR = mr.VecToso3(p[:, 0]) @ R
        assert(pR.shape == (3, 3))

        a = np.block([[R, np.zeros([3, 3])], 
                      [pR, R]])

        assert(a.shape == (6, 6))
        return a


def dist(a, b):
    d = np.linalg.norm(a - b)
    return d

class Puppet:
    """
    • thetalist: an n-vector of initial joint angles (units: rad)
    • dthetalist: an n-vector of initial joint rates (units: rad/s)
    • g: the gravity 3-vector in the {s} frame (units: m/s2)
    • Mlist: the configurations of the link frames relative to each other at the home configuration.
    (There are eight frames total: {0} or {s} at the base of the robot, {1} . . .{6} at the centers
    of mass of the links, and {7} or {b} at the end-effector.)
    • Slist: the screw axes Si in the space frame when the robot is at its home configuration
    • Glist: the spatial inertia matrices Gi of the links (units: kg and kg m2)
    • t: the total simulation time (units: s)
    • dt: the simulation timestep (units: s)
    • damping: a scalar indicating the viscous damping at each joint (units: Nms/rad)
    • stiffness: a scalar indicating the stiffness of the springy string (units: N/m)
    • restLength: a scalar indicating the length of the spring when it is at rest (units: m)
    """
    def __init__(self,
                 dt,
                 Mlist,
                 Slist,
                 Glist,
                 g, 
                 damping = 0.0,
                 stiffness = 0.0,
                 restLength = 0.0,
                 part3 = False,
                 part4 = False
                 ) -> None:
        self.dt = dt
        self.Mlist = Mlist
        self.Slist = Slist
        self.Glist = Glist
        self.g = g
        self.damping = damping
        self.stiffness = stiffness
        self.restLength = restLength
        self.part3 = part3
        self.part4 = part4
        
        self.part3_or_4 = self.part3 or self.part4
        self.M = UR5_M

        # my members
        self.history = defaultdict(list)
        
    def referencePos(self, t):
        if self.part3:
            springPos_s = np.array([1.0, -1.0, 1.0]).T

        elif self.part4:
            pi = np.array([1, 1, 1]).T
            pf = np.array([1, -1, 1]).T

            period = 10.0 / 2.0 # s two full cycles in 10s => period is 5.0s

            # scale time
            t2 = 2 * np.pi *t / period

            # sinusoid, shifted, range [0, 1]
            alpha = np.cos(t2 + np.pi) / 2.0 + 0.5

            # linear interp
            pt = pi * alpha + pf * (1.0 - alpha)

            springPos_s = pt.T

        else:
            springPos_s = np.zeros(3)

        return springPos_s.reshape([3, 1])
    
    def get_B(self, th):
        Tsb = mr.FKinSpace(self.M, self.Slist, th)

        Tsb = Tf(Tsb)

        return Tsb
    
    def get_ee_wrench(self,
                      t,
                      th,
                      Tsb: Tf,
                      ):
        if self.part3_or_4:
            springPos = self.referencePos(t)

            # extract the Bpos
            Bpos = Tsb.getorigin()

            distance = dist(springPos, Bpos)

            # magnitude (stiffness × (distance − restLength)). range [-inf, inf]. Positive = pull towards spring. Negative = push away from spring.
            magnitude = (self.stiffness * (distance - self.restLength))

            # get force vector, final - initial
            fv = Bpos - springPos

            # scale force vector by magnitude
            fv2 = fv * magnitude

            ftip = np.zeros([6, 1])
            ftip[3:6] = fv2
        else:
            ftip = np.zeros([6, 1])

        return ftip

    def forward(self, th, dth, ddth):
        """
        first order euler integration
        """
        thp, dthp = mr.EulerStep(
            th.squeeze(),
            dth.squeeze(),
            ddth.squeeze(),
            self.dt
        )
        return thp.reshape([6, 1]), dthp.reshape([6, 1])
    
    def wrench_space_to_body(self, ftip_s, Tsb: Tf):
        """
        eq 3.95 ftip_b = [Ad_Tab]^T * ftip_a
        ftip_b = [Ad_Tsb]^T * ftip_s
        """
        ad = Tsb.adj()

        ftip_b = ad.T @ ftip_s

        # smoother if I remove the body wrenches
        ftip_b[0:3] = 0.0

        assert(ftip_b.shape == (6, 1))
        return ftip_b
    
    def wrench_to_torque(self, ftips, th):
        """
        eq 5.26
        tau = J^T * ftip
        """
        Js = mr.JacobianSpace(self.Slist, th)

        tau = Js.T @ ftips

        return tau
    
    def get_damping(self, dth):
        """
        tau_d = - self.damping * dth
        """
        tau_d = -self.damping * dth

        assert(tau_d.shape == (6, 1))
        return tau_d

    def run(self,
                thetalist,
                dthetalist,
                tt,
                ):
        """
        outputs:
        • thetamat: an N × n matrix where row i is the set of joint values after simulation step i − 1
        • dthetmat: an N × n matrix where row i is the set of joint rates after simulation step i − 1
        """
        tf = tt
        t = 0.0

        with tqdm.tqdm(total=tf) as pbar:
            while t < tf:
                # forward kinematics
                Tsb = self.get_B(thetalist)

                # get the wrench, space frame
                ftip_s = self.get_ee_wrench(t, thetalist, Tsb)

                # convert ftip to frame {n+1}
                ftip_b = self.wrench_space_to_body(ftip_s, Tsb)

                # get the damping torque
                taulist = self.get_damping(dthetalist)

                # forward dynamics
                ddth = mr.ForwardDynamics(
                    thetalist.squeeze(),
                    dthetalist.squeeze(),
                    taulist.squeeze(),
                    self.g.squeeze(),
                    ftip_b.squeeze(),
                    Mlist,
                    Glist,
                    Slist
                )

                # first order euler integration
                thp, dthp = self.forward(thetalist, dthetalist, ddth)

                # save new state vars
                thetalist = thp
                dthetalist = dthp

                assert(not np.any(dthetalist > 999999.))

                # wrap the joints
                q = thetalist
                q = np.atan2(np.sin(q), np.cos(q))
                thetalist = q

                self.history["th"].append(thetalist)

                t += self.dt
                pbar.update(self.dt)

    def save(self, prefix):
        qh = np.array(self.history["th"]).squeeze()
        np.savetxt("./{}.csv".format(prefix), qh, delimiter=',')


################# UR5 params #################
M01 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]]
M12 = [[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0], [0, 0, 0, 1]]
M23 = [[1, 0, 0, 0], [0, 1, 0, -0.1197], [0, 0, 1, 0.395], [0, 0, 0, 1]]
M34 = [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.14225], [0, 0, 0, 1]]
M45 = [[1, 0, 0, 0], [0, 1, 0, 0.093], [0, 0, 1, 0], [0, 0, 0, 1]]
M56 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.09465], [0, 0, 0, 1]]
M67 = [[1, 0, 0, 0], [0, 0, 1, 0.0823], [0, -1, 0, 0], [0, 0, 0, 1]]
G1 = np.diag([0.010267495893, 0.010267495893,  0.00666, 3.7, 3.7, 3.7])
G2 = np.diag([0.22689067591, 0.22689067591, 0.0151074, 8.393, 8.393, 8.393])
G3 = np.diag([0.049443313556, 0.049443313556, 0.004095, 2.275, 2.275, 2.275])
G4 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
G5 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
G6 = np.diag([0.0171364731454, 0.0171364731454, 0.033822, 0.1879, 0.1879, 0.1879])
Glist = [G1, G2, G3, G4, G5, G6]
Mlist = [M01, M12, M23, M34, M45, M56, M67] 
Slist = [[0,         0,         0,         0,        0,        0],
         [0,         1,         1,         1,        0,        1],
         [1,         0,         0,         0,       -1,        0],
         [0, -0.089159, -0.089159, -0.089159, -0.10915, 0.005491],
         [0,         0,         0,         0,  0.81725,        0],
         [0,         0,     0.425,   0.81725,        0,  0.81725]]
#############################################

def q1():
    # part 1
    g = np.array([0, 0, -9.81]).T
    tt = 10.0
    dts = [0.005, 0.5]
    names = ["part1a", "part1b"]
    th = np.zeros([6, 1])
    dth = np.zeros_like(th)

    for i in range(2):
        dt = dts[i]
        puppeteer = Puppet(dt, Mlist, Slist, Glist, g)
        puppeteer.run(th, dth, tt)

        # save csv
        puppeteer.save(names[i])

def q2():
    # part 2
    g = np.array([0, 0, -9.81]).T
    tt = 10.0
    dt = 0.01
    th = np.zeros([6, 1])
    dth = np.zeros_like(th)
    dampings = [2.0, -0.01]
    names = ["part2a", "part2b"]

    for i in range(2):
        damping = dampings[i]
        puppeteer = Puppet(dt, Mlist, Slist, Glist, g, damping)
        puppeteer.run(th, dth, tt)

        # save csv
        puppeteer.save(names[i])

def q3():
    # part 3
    g = np.zeros([1, 3])
    tt = 10.0
    dt = 0.01
    damping = 0.0
    restLength = 0.0
    th = np.zeros([6, 1])
    dth = np.zeros_like(th)
    stiffnesss = [5.0] # , 10.0]

    # part 3a
    if False:
        for stiffness in stiffnesss:
            puppeteer = Puppet(dt, Mlist, Slist, Glist, g, damping, stiffness, restLength, part3=True)
            puppeteer.run(th, dth, tt)

            puppeteer.save("part3a")

    # part 3b
    if True:
        damping = 2.0
        stiffness = 5.0
        puppeteer = Puppet(dt, Mlist, Slist, Glist, g, damping, stiffness, restLength, part3=True)
        puppeteer.run(th, dth, tt)
        puppeteer.save("part3b")

def q4():
    # part 4
    tt = 10.0
    dt = 0.01
    damping = 2.0
    stiffness = 5.0
    g = np.zeros([1, 3])
    restLength = 0.0
    th = np.zeros([6, 1])
    dth = np.zeros_like(th)

    puppeteer = Puppet(dt, Mlist, Slist, Glist, g, damping, stiffness, restLength, part4=True)
    puppeteer.run(th, dth, tt)
    puppeteer.save("part4")

def main():
    # q1()

    # q2()

    # q3()

    q4()

if __name__ == "__main__":
    main()
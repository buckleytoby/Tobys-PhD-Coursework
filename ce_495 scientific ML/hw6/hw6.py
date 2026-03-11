


import numpy as np
import sympy
from sympy import *

import skimage
import skimage.data

import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import scipy.integrate as integrate



### Problem 1
def f(x):
    # 1/(x-1)^(1/4)
    y = 1.0 / ((x-1.0)**(1/4))
    return y


def get_error(z):
    z_exact = 3.2 - 0.94*sympy.I
    error_re = (re(z) - re(z_exact))**2
    error_im = (im(z) - im(z_exact))**2

    # 4. Sum them up and divide by 2 for the MSE
    mse = (error_re + error_im) / 2

    return mse

class RiemannSums:
    def __init__(self, n):
        self.n = n

        self.calc_midpoint_ck()

    def calc_midpoint_ck(self):
        x = sympy.Symbol('x')
        expr = 1.0 / ((x-1.0)**(1/4))

        xs = np.linspace(0.0, 3.0, self.n)
        dx = np.diff(xs)

        cks = (xs[1:] + xs[0:-1]) / 2

        # f = sympy.lambdify(x, expr, modules="numpy")
        # fcks = f(cks)
        total = 0.0
        for i in range(len(cks)):
            ck = cks[i]
            fck = expr.subs(x, ck)
            area = fck * dx[i]

            total += area

        farea = total.evalf()
        error = get_error(farea)

        print(f"calc_midpoint_ck n = {self.n}, area = {farea}, error = {error}")
        return farea

    def calc_ck_equals_xk(self):
        x = sympy.Symbol('x')
        expr = 1.0 / ((x-1.0)**(1/4))

        xs = np.linspace(0.0, 3.0, self.n)
        dx = np.diff(xs)

        cks = xs[0:-1]

        # f = sympy.lambdify(x, expr, modules="numpy")
        # fcks = f(cks)
        total = 0.0
        for i in range(len(cks)):
            ck = cks[i]
            fck = expr.subs(x, ck)
            area = fck * dx[i]

            total += area

        farea = total.evalf()
        error = get_error(farea)

        print(f"calc_ck_equals_xk n = {self.n}, area = {farea}, error = {error}")
        return farea
    def calc_ck_equals_xk_plus_1(self):
        x = sympy.Symbol('x')
        expr = 1.0 / ((x-1.0)**(1/4))

        xs = np.linspace(0.0, 3.0, self.n)
        dx = np.diff(xs)

        cks = xs[1:]

        # f = sympy.lambdify(x, expr, modules="numpy")
        # fcks = f(cks)
        total = 0.0
        for i in range(len(cks)):
            ck = cks[i]
            fck = expr.subs(x, ck)
            area = fck * dx[i]

            total += area

        farea = total.evalf()
        error = get_error(farea)

        print(f"calc_ck_equals_xk_plus_1 n = {self.n}, area = {farea}, error = {error}")
        return farea


class Problem1:
    def __init__(self):
        print("Problem 1")
        self.partc()
        self.partd()

    def partc(self):
        print("part c")
        N = [3, 7, 15]

        for n in N:
            rms = RiemannSums(n)

    def partd(self):
        print("part d")

        rms = RiemannSums(500)
        rms.calc_ck_equals_xk()
        rms.calc_ck_equals_xk_plus_1()


def check_operator_singularity(L, tol=1e-10):
    # 1. Get the eigenvalues of L
    eigvals = np.linalg.eigvals(L)
    
    # 2. Use broadcasting to create a matrix of all possible sums
    # eigvals[:, None] is a column vector, eigvals[None, :] is a row vector
    sums = eigvals[:, np.newaxis] + eigvals[np.newaxis, :]
    
    # 3. Check if any sum is effectively zero
    zero_sums = np.where(np.abs(sums) < tol)
    indices = list(zip(zero_sums[0], zero_sums[1]))
    
    is_surjective = len(indices) == 0
    
    return is_surjective, indices, eigvals

class Problem3:
    def __init__(self):
        print("Problem 3")
        self.partc()

    def partc(self):
        cameraman = skimage.data.camera()

        shape = cameraman.shape
        print("shape: ", shape)

        # plot the image
        plt.imshow(cameraman, cmap='gray')
        plt.show()

        # setup L
        L = -2.0 * np.identity(shape[0])
        for i in range(shape[0]):
            if i > 0:
                L[i, i-1] = 1.0
            if i < shape[0]-1:
                L[i, i+1] = 1.0

        print(L)

        # get T
        T = L@cameraman + cameraman@L

        I1 = np.zeros(shape)
        I1[1,2] = 1.0
        I2 = np.zeros(shape)
        I2[0,1] = -1.0

        # plot T
        plt.imshow(T, cmap='gray')
        plt.show()

        print("rank: ", np.linalg.matrix_rank(T))
        print("inverse: ", np.linalg.inv(T))

        L2 = np.linalg.inv(L)
        T3 = L2@T

        plt.imshow(T3, cmap='gray')
        plt.show()




class Problem5:
    def __init__(self):
        print("Problem 5")

        g = 9.81
        
        def straightline(x):
            return -x
        def pstraightline(x):
            return -1.0
        
        def quad(x):
            return (x-1)**2-1
        def pquad(x):
            return 2*(x-1)
        

        x = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
        y = np.array([0, -0.1498, -0.32354, -0.4543, -0.8345234, -1])
        cs = CubicSpline(x, y, bc_type='natural') 
        def spline(x):
            return cs(x)
        def pspline(x):
            return cs(x, 1)
            
        # https://www.myphysicslab.com/roller/brachistochrone-en.html
        # x2=−a(θ−sinθ)+x1
        # y2=−a(1−cosθ)+y1
        # (0, 0), (1, -1)
        # 1 = -a(θ-sinθ)
        # -1 = -a(1-cosθ)
        # theta ~ 2.412
        # a ~ 0.572
        # a = 0.572
        finaltheta = -2.412

        def brachistochrone_theta(x):
            a = 0.572
            # x=−a(θ−sinθ)+x1
            # solve for theta given x
            thetas = sympy.Symbol('theta')
            eq = sympy.Eq(-a*(thetas-sympy.sin(thetas)), x)
            theta = float(sympy.nsolve(eq, thetas, finaltheta))

            return theta
        
        def brachistochrone_x(theta):
            a = 0.572
            # x=−a(θ−sinθ)+x1
            return -a*(theta-np.sin(theta))

        def brachistochrone(theta):
            a = 0.572
            # x=−a(θ−sinθ)+x1
            # solve for theta given x
            # xs = sympy.Symbol('x')
            # thetas = sympy.Symbol('theta')
            # eq = sympy.Eq(-a*(thetas-sympy.sin(thetas)), x)
            # theta = float(sympy.nsolve(eq, thetas, finaltheta))

            return -a*(1-np.cos(theta))
        def pbrachistochrone(theta):
            # a = 0.572
            # y = brachistochrone(x)
            
            # yp = np.sqrt((2*a-y)/y)
            # return yp
            # cot(t/2)
            return 1/np.tan(theta/2)
        
        """
        a, b = sp.symbols('a b')

        # Equation 1: 1/(1+exp(b)) + a = 0
        eq1 = 1/(1 + sp.exp(b)) + a

        # Equation 2: -1 = a + 1/(1+exp(-(1+b)))
        # Assuming the dot symbol represents a negative sign based on common mathematical contexts for sigmoid functions
        eq2 = a + 1/(1 + sp.exp(-(1 + b))) + 1

        sol = sp.solve((eq1, eq2), (a, b))
        print(sol)
        """

        def sigmoid(x):
            # a = -4.083
            # b = 1.541
            a= -1.01
            b=0.0068
            return a/(1+np.exp(-10*(x-0.5)))+b
        def psigmoid(x):
            a = -1.01
            exponent = -10 * (x - 0.5)
            denominator = (1 + np.exp(exponent))**2
            
            v= (10 * a * np.exp(exponent)) / denominator
            return v


        def L(x, fcn, pfcn):
            """
            y = [y, yp]
            """
            y = fcn(x)
            yp = pfcn(x)
            return np.sqrt((1+yp**2)/(1e-6 + 2*g*(np.abs(y))))
        
        def L2(theta):
            y = brachistochrone(theta)
            yp = pbrachistochrone(theta)
            return np.sqrt((1+yp**2)/(2*g*(np.abs(y))))
        

        x = np.linspace(0, 1, 1000)
        y = straightline(x)
        plt.plot(x, y, label="straightline")
        plt.title("Straight Line")
        plt.show()
        y = quad(x)
        plt.plot(x, y, label="quad")
        plt.title("Quadratic")
        plt.show()
        y = spline(x)
        plt.plot(x, y, label="spline")
        plt.title("Spline")
        plt.show()
        y = sigmoid(x)
        plt.plot(x, y, label="sigmoid")
        plt.title("Sigmoid")
        plt.show()





        

        print(f"Solution time straightline: {integrate.quad(L, 0, 1, args=(straightline, pstraightline))[0]}")
        print(f"Solution time quad: {integrate.quad(L, 0, 1, args=(quad, pquad))[0]}")
        print(f"Solution time spline: {integrate.quad(L, 0, 1, args=(spline, pspline))[0]}")
        print(f"Solution time sigmoid: {integrate.quad(L, 0, 1, args=(sigmoid, psigmoid))[0]}")

        n = 100000000
        theta = np.linspace(-0.0000001, finaltheta, n)
        # theta = brachistochrone_theta(x)
        x = brachistochrone_x(theta)
        y = brachistochrone(theta)
        yp = pbrachistochrone(theta)
        Ls = L2(theta)
        plt.plot(x, y, label="brachistochrone")
        plt.title("Brachistochrone")
        plt.show()
        print(f"Solution time brachistochrone: {integrate.trapezoid(Ls, x)}")



if __name__ == "__main__":
    p1 = Problem1()
    p3 = Problem3()
    p5 = Problem5()
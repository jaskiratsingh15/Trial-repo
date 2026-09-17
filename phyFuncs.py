"""
Quantum Mechanics Computational Lab Assignment 5 - 2024PHY1005 - Jaskirat Singh
"""

import numpy as np
import matplotlib.pyplot as plt


def lsf(x: np.ndarray | list,
        y: np.ndarray | list) -> tuple | None:
    """
    Returns least square fit parameters: mean, std_dev, slope and intercept

    Inputs:
    x: values of independent variable
    y: values of dependent variable

    Outputs:
    params: calculated parameters - mean, std_dev, slope and intercept
    """
    if len(x)!=len(y):
        return(None)
    
    if type(x) == list:
        x = np.asarray(x)
    if type(y) == list:
        y = np.asarray(y)

    n = len(x)
    sum_x    = np.sum(x)
    sum_y    = np.sum(y)
    sum_xy   = np.sum(x*y)
    sum_x_sq = np.sum(x**2)

    mean                = sum_y/n
    # std_dev             = np.sqrt(np.sum((mean-y)**2)/n)
    numerator_slope     = (n*sum_xy) - (sum_x*sum_y)
    numerator_intercept = (sum_y*sum_x_sq) - (sum_xy*sum_x)
    denominator         = (n*sum_x_sq) - (sum_x**2)
    slope               = numerator_slope/denominator
    intercept           = numerator_intercept/denominator   #(mean) - (slope*(np.sum(x)/n))
    y_best = slope*x + intercept
    res = y_best - y
    res_mean = np.sum(res)/n
    std_dev = np.sqrt(np.sum((res_mean-res)**2)/n)

    params = (mean, std_dev, slope, intercept)
    return(params)

def my_bisection(func:function,
                 a:float,
                 b:float,
                 max_iters: int = 1000,
                 tol: float = 1e-6,
                 *args, **kwargs) -> float | str:
    """
    Inputs:
    func: function for f(x);
    a: lower limit;
    b: upper limit;
    max_iters: maximum number of iterations;
    tol: tolerance

    Output:
    msg: string telling that accuracy could not be achieved within the max iters;
    pn: nth approximation to the root
    """
    f   = func
    err = abs(a-b)
    pn  = (a+b)/2  #initializing the root at 0th iteration

    if (f(a, *args, **kwargs)>0 and f(b, *args, **kwargs)>0) or (f(a, *args, **kwargs)<0 and f(b, *args, **kwargs)<0):
        #the root can be outside the range or multiple roots exits in the range
        msg = "f(a) and f(b) must have different signs"
        return(msg)

    iterations = 0

    while (err>=tol and iterations<=max_iters):
        iterations += 1
        fa  = f(a,  *args, **kwargs)
        fb  = f(b,  *args, **kwargs)
        fpn = f(pn, *args, **kwargs)
        if (fa*fpn > 0):
            a = pn

        elif (fb*fpn > 0):
            b = pn

        elif (fa==0 or fb==0 or fpn==0):
            return(a if fa==0 else (b if fb==0 else pn), iterations)
        
        else:
            return(None)
        
        err = abs(a-b)
        pn  = (a+b)/2

    if (err>tol and iterations>=max_iters):
        msg = "Tolerance could not be achieved within the maximum number of iterations"
        return(msg)
    
    return(pn, iterations)

def my_secant(func:function,
              a:float,
              b:float,
              max_iters: int = 1000,
              tol: float = 1e-6,
              *args, **kwargs) -> float | str:
    """
    Inputs:
    func: function for f(x);
    a: lower limit;
    b: upper limit;
    max_iters: maximum number of iterations;
    tol: tolerance

    Output:
    msg: string telling that accuracy could not be achieved within the max iters;
    pn: nth approximation to the root
    """
    f    = func
    pn_1 = a
    pn   = b
    err  = abs(pn - pn_1)
    iterations = 0

    while (err <= tol and iterations<=max_iters):
        iterations += 1
        fpn_1     = f(pn_1, *args, **kwargs)
        fpn       = f(pn  , *args, **kwargs)
        pn_plus_1 = ((pn_1*fpn) - (pn*fpn_1))/(fpn - fpn_1)

        err  = abs(pn_plus_1 - pn)
        pn_1 = pn
        pn   = pn_plus_1

    if (err>tol and iterations>=max_iters):
        msg = "Tolerance could not be achieved within the maximum number of iterations"
        return(msg)

    return(pn, iterations)

def my_NR(func:function,
          func_derv:function,
          a:float,
          b:float,
          max_iters: int = 1000,
          tol: float = 1e-6,
          *args, **kwargs) -> float | str:
    """
    Inputs:
    func: function for f(x);
    func_derv: derivative for f(x);
    a: lower limit;
    b: upper limit;
    max_iters: maximum number of iterations;
    tol: tolerance

    Output:
    msg: string telling that accuracy could not be achieved within the max iters;
    pn: nth approximation to the root
    """
    f   = func
    Df = func_derv
    pn  = (b-a)/2
    err = abs(pn - a) #error of pn from a shall be same as that from b
    iterations = 0

    while (err <= tol and iterations<=max_iters):
        iterations += 1
        fpn       = f(pn, *args, **kwargs)
        f_d_pn    = Df(pn, *args, **kwargs)
        pn_plus_1 = pn - (fpn/f_d_pn)

        err = abs(pn_plus_1 - pn)
        pn  = pn_plus_1

    if (err>tol and iterations>=max_iters):
        msg = "Tolerance could not be achieved within the maximum number of iterations"
        return(msg)

    return(pn, iterations)

def riemann(func: function | list | np.ndarray,
            a:float,
            b:float,
            n = 1000,
            *args, **kwargs) -> float:
    """
    func: function for f(x);
    a: lower limit;
    b: upper limit;
    n: number of steps

    Outputs
    integrate_sum: array of integrated values
    """

    integrate_sum = 0

    if callable(func):
        h = (abs(a-b)/n) #step size
        x = np.linspace(a, b, n)
        integrate_sum = np.sum(func(x=x, *args, **kwargs))*h

    else:
        y = np.asarray(func)
        n = len(y)
        h = abs(b-a)/(n-1)
        integrate_sum = np.sum(y[:-1])*h

    return(integrate_sum)

def trapezoidal(func: function | list | np.ndarray,
                a:float,
                b:float,
                n=1000,
                *args, **kwargs) -> float:
    """
    func: function for f(x);
    a: lower limit;
    b: upper limit;
    n: number of steps

    Outputs
    integrate_sum: array of integrated values
    """
    
    integrate_sum = 0

    if callable(func):
        h = abs(b-a)/n
        x = np.linspace(a, b, n+1)
        y = func(x=x, *args, **kwargs)

        integrate_sum = (y[0] + y[-1]) + 2*(np.sum(y[1:-1]))
        integrate_sum *= h/2

    else:
        y = np.asarray(func)
        n = len(y)
        h = abs(b-a)/(n-1)

        integrate_sum = (y[0] + y[-1]) + 2*(np.sum(y[1:-1]))
        integrate_sum *= h/2

    return(integrate_sum)

def simpson_1_3(func: function | list | np.ndarray,
                a:float,
                b:float,
                n=1000,
                *args, **kwargs) -> float:
    """
    func: function for f(x);
    a: lower limit;
    b: upper limit;
    n: number of steps

    Outputs
    integrate_sum: array of integrated values
    """

    if callable(func):
        if n%2!=0:
            n -= 1 #num of intervals are even
        h = abs(b-a)/n
        x = np.linspace(a, b, n+1)
        y = func(x, *args, **kwargs)

    else:
        y = np.asarray(func)
        n = len(y)
        if n%2 == 0:
            y = y[:-1] #left the last point to fulfill the even intervals condition required by simpson's 1/3
            n = len(y)
        h = abs(b-a)/(n-1)

    integrate_sum = (y[0] + y[-1]) + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2])
    integrate_sum *= h/3

    return(integrate_sum)

def simpson_3_8(*args, **kwargs):
    print("Simpson's 3/8 Method Under Construction")
    return(None)

def my_euler(func:list,
             a:float,
             b:float,
             alpha:list,
             n:int = 100,
             *args, **kwargs) -> np.ndarray:
    """
    Inputs
    func: expressions, packed inside a list, for the derivatives in sys of linear DE;
    a: lower limit of solving;
    b: upper limit of solving;
    alpha: list containing initial values of the sys of linear DE;
    n: number of intervals

    Outputs
    x: array of x values
    u: m X (n+1) size array of solutions
    """
    h = (b-a)/n
    m = len(func)
    x = np.linspace(a, b, n+1)

    if len(func)!=len(alpha):
        return(None)
    
    #u is the array of approximate solutions
    u = [[a] for a in alpha] #initializing with initial conditions at ti = a
    
    #i shall represent indexing of steps, ranging from 1 to n
    #j shall represent indexing of number of DE, ranging 0 to m-1

    #i is the column number in u
    #j is the row number in u

    for i in range(1,n+1):
        ti = a + i*h
        for j in range(m):
            u_j_i = u[j][i-1] + h*func[j](ti-h,[row[-1] for row in u], *args, **kwargs)
            u[j].append(u_j_i)

    return(x, np.array(u))

def my_RK2(func:list,
           a:float,
           b:float,
           alpha:list,
           n:int = 100,
           *args, **kwargs) -> np.ndarray:
    """
    Inputs
    func: expressions, packed inside a list, for the derivatives in sys of linear DE;
    a: lower limit of solving;
    b: upper limit of solving;
    alpha: list containing initial values of the sys of linear DE;
    n: number of intervals

    Outputs
    x: array of x values
    u: m X (n+1) size array of solutions
    """
    h = (b-a)/n
    m = len(func)
    x = np.linspace(a, b, n+1)

    #u is the array of approximate solutions, and ks are the intermediate approximate solutions
    u = [[a] for a in alpha]; k1 = [[] for i in range(m)]; k2 = [[] for i in range(m)]
    
    #i shall represent indexing of steps, ranging from 1 to n
    #j shall represent indexing of number of DE, ranging 0 to m-1

    #i is the column number in u and k1 and k2
    #j is the row number in u and k1 and k2

    for i in range(1,n+1):
        ti = a + i*h
        for j in range(m):
            for k_idx in range(m):
                #calculating k1 for first approximation
                k_j_i = h*func[j](ti-h, [row[-1] for row in u], *args, **kwargs)
                k1[k_idx].append(k_j_i)

            #calculating k2 for second approximation at next step
            u_tilda = [row_u[-1]+row_k1[-1] for row_u, row_k1 in zip(u,k1)]

            for k_idx in range(m):
                k_j_i = h*func[j](ti, u_tilda, *args, **kwargs)
                k2[k_idx].append(k_j_i)

            #calculating the approximated solution at (i+1)
            u_j_i = u[j][i-1] + (k1[j][-1] + k2[j][-1])/2
            u[j].append(u_j_i)

    return(x, np.array(u))

def my_RK4(func:list,
           a:float,
           b:float,
           alpha:list,
           n:int = 100,
           *args, **kwargs) -> np.ndarray:
    """
    Inputs
    func: expressions, packed inside a list, for the derivatives in sys of linear DE;
    a: lower limit of solving;
    b: upper limit of solving;
    alpha: list containing initial values of the sys of linear DE;
    n: number of intervals

    Outputs
    x: array of x values
    u: m X (n+1) size array of solutions
    """
    h = (b-a)/n
    m = len(func)
    x = np.linspace(a, b, n+1)
    
    #u is the array of approximate solutions, and ks are the intermediate approximate solutions
    u = [[a] for a in alpha]; k1 = [[] for i in range(m)]; k2 = [[] for i in range(m)]; k3 = [[] for i in range(m)]; k4 = [[] for i in range(m)]
    
    #i shall represent indexing of steps, ranging from 1 to n
    #j shall represent indexing of number of DE, ranging 0 to m-1

    #i is the column number in u and k1 and k2 and k3 and k4
    #j is the row number in u and k1 and k2 and k3 and k4

    for i in range(1,n+1):
        ti = a + i*h
        for j in range(m):
            for k_idx in range(m):
                #calculating k1 for first approximation
                k_j_i = h*func[j](ti-h, [row[-1] for row in u], *args, **kwargs)
                k1[k_idx].append(k_j_i)

            #calculating k2 for second approximation at next half step
            u_tilda_approx_2 = [row_u[-1]+row_k1[-1] for row_u, row_k1 in zip(u,k1)]

            for k_idx in range(m):
                k_j_i = h*func[j](ti-(h/2), u_tilda_approx_2, *args, **kwargs)
                k2[k_idx].append(k_j_i)

            #calculating k3 for second approximation at next half step
            u_tilda_approx_3 = [row_u[-1]+(0.5*row_k2[-1]) for row_u, row_k2 in zip(u,k2)]

            for k_idx in range(m):
                k_j_i = h*func[j](ti-(h/2), u_tilda_approx_3, *args, **kwargs)
                k3[k_idx].append(k_j_i)

            #calculating k4 for second approximation at next step
            u_tilda_approx_4 = [row_u[-1]+(0.5*row_k3[-1]) for row_u, row_k3 in zip(u,k3)]

            for k_idx in range(m):
                k_j_i = h*func[j](ti, u_tilda_approx_4, *args, **kwargs)
                k4[k_idx].append(k_j_i)

            #calculating the approximated solution at (i+1)
            u_j_i = u[j][i-1] + (k1[j][-1] + 2*k2[j][-1] + 2*k3[j][-1] + k4[j][-1])/6
            u[j].append(u_j_i)

    return(x, np.array(u))

def shooting(func: list,
             a: float,
             b: float,
             alpha: list,
             u_at_right_boundary,
             x: list,
             y: list,
             n: int = 100,
             tol: float = 1e-3,
             *args, **kwargs) -> float:
    """
    Uses Secant as the root finder and RK4 as the ODE solver

    Inputs:
    func: list of linear ODEs;
    a: lower limit of solving;
    b: upper limit of solving;
    alpha: list of initial conditions;
    u_at_right_boundary: boundary condition for he dependent variable at the other end of the domain;
    x: list of two independent variables for which the residual changes sign;
    y: list of residuals corresponding to independent variables in list x;
    n: number of intervals;
    tol: tolerance in residual

    Outputs:
    epsilon: the best value for energy
    """
    #first find next approximation using a root finder and then determine the value of residue at that point using RK4
    x_a, x_b = x
    f_a, f_b = y

    x_new = (x_a*f_b - x_b*f_a)/(f_b-f_a) #using secant method

    f_new = my_RK4(func = func, a = a, b = b, alpha = alpha, n = n, epsilon = x_new)[1][0][-1] - u_at_right_boundary

    iterations = 1

    while f_new>=tol:
        iterations += 1
        x_a = x_b; x_b = x_new; f_a = f_b; f_b = f_new
        x_new = (x_a*f_b - x_b*f_a)/(f_b-f_a)
        f_new = my_RK4(func, a, b, alpha, n, epsilon = x_new, *args, **kwargs)[1][0][-1] - u_at_right_boundary

    return(x_new)
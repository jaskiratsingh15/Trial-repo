import numpy as np
from phyFuncs import *
import os
from tabulate import tabulate
from scipy.optimize import newton, bisect

#function for saving terminal output
def file_write(output_file, output_list):
    output_file.writelines(output_list)
    for line in output_list: print(line, end='')

#f(x) = tan(x)-x
def func1(x: list | np.ndarray | float | int) -> list | np.ndarray | float | int:
    if type(func1) == list:
        x = np.asarray(x)
    return(np.tan(x)-x)

#f'(x) = sec(x)^2 - 1 = tan(x)^2
def func1_derv(x: list | np.ndarray | float | int) -> list | np.ndarray | float | int:
    if type(func1) == list:
        x = np.asarray(x)
    return(np.tan(x)**2)

#nth root of a number M
def func2(x: list | np.ndarray | float | int,
          n:float | int = 1,
          M: float | int = 0) -> list | np.ndarray | float | int:
    if type(func1) == list:
        x = np.asarray(x)
    
    return(x**n - M)

#derivative of nth root of a number M
def func2_derv(x: list | np.ndarray | float | int,
          n:float | int = 1,
          M: float | int = 0) -> list | np.ndarray | float | int:
    if type(func1) == list:
        x = np.asarray(x)
    
    return(n * (x**(n-1)))

def plt_tan_func():
    #plotting the function in a wide interval
    ll = -4*np.pi; ul = 4*np.pi
    x = np.linspace(ll, ul, 1000)
    y = func1(x)

    fig, ax = plt.subplots(1, 1, figsize = (12,6))

    ax.plot(x, y)
    ax.set_ylim(-20,20)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f(x)$")
    ax.set_title(r"$f(x) = \tan(x) - x$")
    ax.grid(ls='--', lw = 0.75, alpha = 0.6, c='grey')
    ax.axhline(0, c='black', lw=0.75)
    ax.axvline(0, c='black', lw=0.75)
    plt.tight_layout()
    fig_name = 'fx-tan-in-wide-interval'
    plt.savefig(f"plots/{fig_name}.png")
    plt.savefig(f"plots/{fig_name}.pdf")
    file_write(output_file, [f"\nPlot saved as {fig_name}.png and {fig_name}.pdf in 'plots' directory\n\n"])
    plt.show()

def tan_func_root_find():
    plt_tan_func()
    #solving numerically in the chosen interval
    file_write(output_file, ["\n---------Choose the upper and lower limits for root finding below---------\n\n"])
    a_di = float(input("Enter lower limit of solution interval: "))
    b_di = float(input("Enter upper limit of solution interval: "))
    file_write(output_file, [f"\nChosen interval: [{a_di},{b_di}]\n\n"])
    # max_iters_di = 10
    fx_di = "tan(x) - x" #to display
    output = my_bisection(func=func1, a=a_di, b=b_di) #, max_iters=max_iters_di)


    if type(output) == str:
        file_write(output_file, ['\n\n'+output+'\n\n'])
    else:
        root_bisection, iterations = output

        scipy_root_bisect_di = bisect(lambda x: np.tan(x)-x, a_di, b_di)
        x0 = (a_di+b_di)/2
        scipy_root_secant_di = newton(lambda x: np.tan(x)-x, x0 = x0)
        scipy_root_NR_di = newton(lambda x: np.tan(x)-x, fprime = lambda x: np.tan(x)**2, x0 = x0)


        dict_di = {"Type of Function":["Defined", "In-built"],
                "Rounded off Root":[root_bisection, scipy_root_bisect_di]}
        table = tabulate(dict_di, headers='keys', tablefmt='grid')
        file_write(output_file, [f"f(x) = {fx_di}\nMethod: Bisection\nInterval: [{a_di},{b_di}]\nIterations for In-built method: {iterations}\n",
                                table+'\n\n'])

        root_secant, iterations = my_secant(func=func1, a=a_di, b=b_di)
        dict_di = {"Type of Function":["Defined", "In-built"],
                "Rounded off Root":[root_secant, scipy_root_secant_di]}
        table = tabulate(dict_di, headers='keys', tablefmt='grid')
        file_write(output_file, [f"f(x) = {fx_di}\nMethod: Secant\nInterval: [{a_di},{b_di}]\nIterations for In-built method: {iterations}\n",
                                table+'\n\n'])

        root_NR, iterations = my_NR(func=func1, func_derv = func1_derv, a=a_di, b=b_di)
        dict_di = {"Type of Function":["Defined", "In-built"],
                "Rounded off Root":[root_NR, scipy_root_NR_di]}
        table = tabulate(dict_di, headers='keys', tablefmt='grid')
        file_write(output_file, [f"f(x) = {fx_di}\nMethod: Newton-Raphson\nInterval: [{a_di},{b_di}]\nIterations for In-built method: {iterations}\n",
                                table+'\n\n'])

        #plotting the function in sol interval
        x = np.linspace(a_di, b_di, 1000)
        y = func1(x)

        root = (root_bisection + root_secant + root_bisection) / 3

        fig, ax = plt.subplots(1, 1,figsize=(6,6))

        ax.plot(x,y)
        ax.set_ylim(-10,10)
        ax.axvline(root, ls='--', c='r', label=f"Numerical root={root:.3f}")
        ax.axhline(0, c='black', lw=0.75)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$f(x)$")
        ax.set_title(fr"$f(x) = \tan(x) - x \ $, in selected interval [{a_di},{b_di}]")
        ax.grid(ls='--', lw = 0.75, alpha = 0.6, c='grey')
        ax.legend()
        plt.tight_layout()
        fig_name = f'fx-tan-in-sol-interval-{a_di}-to-{b_di}'
        plt.savefig(f"plots/{fig_name}.png")
        plt.savefig(f"plots/{fig_name}.pdf")
        plt.show()
        file_write(output_file, [f"\nPlot saved as {fig_name}.png and {fig_name}.pdf in 'plots' directory\n\n"])

def plt_nth_root_func(n_dii, M_dii):
    #plotting the function in a wide interval
    x = np.linspace(-M_dii, M_dii, 1000)
    y = func2(x, n=n_dii, M=M_dii)

    fig, ax = plt.subplots(1, 1, figsize = (6, 6))

    ax.plot(x, y)
    ax.axhline(0, lw=0.75, c='black')
    ax.axvline(0, lw=0.75, c='black')
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f(x)$")
    ax.set_title(fr"$f(x) = x^{{{n_dii}}} - {M_dii}$")
    ax.grid(ls='--', lw = 0.75, alpha = 0.6, c='grey')
    plt.tight_layout()
    fig_name = 'fx-nth-root-in-wide-interval'
    plt.savefig(f"plots/{fig_name}.png")
    plt.savefig(f"plots/{fig_name}.pdf")
    plt.show()
    file_write(output_file, [f"\nPlot saved as {fig_name}.png and {fig_name}.pdf in 'plots' directory\n\n"])

def nth_root_func_root_find():
    n_dii = float(input("Which root to find: "))
    M_dii = float(input(f"Enter the number to find {n_dii}th root: "))

    plt_nth_root_func(n_dii, M_dii)

    #solving numerically in the chosen interval
    file_write(output_file, ["\n---------Choose the upper and lower limits for root finding below---------\n\n"])

    a_dii = float(input(f"Enter lower limit of solution interval: "))
    b_dii = float(input(f"Enter upper limit of solution interval: "))
    file_write(output_file, [f"\nChosen interval: [{a_dii},{b_dii}]\n\n"])
    x_dii = np.linspace(a_dii, b_dii, 1000)
    y_dii = func2(x_dii, n_dii, M_dii)
    fx_dii = f"x^({n_dii}) - {M_dii}" #to display


    #max_iters_dii = 10
    output = my_bisection(func=func2, a=a_dii, b=b_dii, n = n_dii, M = M_dii) #, max_iters=max_iters_dii)


    if type(output) == str:
        file_write(output_file, [output])
    else:
        root_bisection, iterations = output
        scipy_root_bisect_di = bisect(lambda x: x**n_dii - M_dii, a_dii, b_dii)
        x0 = (a_dii + b_dii) / 2
        scipy_root_secant_di = newton(lambda x: x**n_dii - M_dii, x0 = x0)
        scipy_root_NR_di = newton(lambda x: x**n_dii - M_dii, fprime = lambda x: n_dii * (x**(n_dii - 1)), x0 = x0)

        dict_di = {"Type of Function":["Defined", "In-built"],
                "Rounded off Root":[root_bisection, scipy_root_bisect_di]}
        table = tabulate(dict_di, headers='keys', tablefmt='grid')
        file_write(output_file, [f"f(x) = {fx_dii}\nMethod: Bisection\nInterval: [{a_dii},{b_dii}]\nIterations for In-built method: {iterations}\n",
                                table+'\n\n'])

        root_secant, iterations = my_secant(func=func2, a=a_dii, b=b_dii, n = n_dii, M = M_dii)
        dict_di = {"Type of Function":["Defined", "In-built"],
                "Rounded off Root":[root_secant, scipy_root_secant_di]}
        table = tabulate(dict_di, headers='keys', tablefmt='grid')
        file_write(output_file, [f"f(x) = {fx_dii}\nMethod: Secant\nInterval: [{a_dii},{b_dii}]\nIterations for In-built method: {iterations}\n",
                                table+'\n\n'])

        root_NR, iterations = my_NR(func=func2, func_derv = func2_derv, a=a_dii, b=b_dii, n = n_dii, M = M_dii)
        dict_di = {"Type of Function":["Defined", "In-built"],
                "Rounded off Root":[root_NR, scipy_root_NR_di]}
        table = tabulate(dict_di, headers='keys', tablefmt='grid')
        file_write(output_file, [f"f(x) = {fx_dii}\nMethod: Newton-Raphson\nInterval: [{a_dii},{b_dii}]\nIterations for In-built method: {iterations}\n",
                                table+'\n\n'])
        
        root = (root_bisection + root_secant + root_bisection) / 3
        
        fig, ax = plt.subplots(1, 1, figsize = (6, 6))

        ax.plot(x_dii, y_dii)
        ax.axvline(root, ls='--', c='r', label=f"Numerical root={root:.3f}")
        ax.axhline(0, c='black', lw=0.75)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$f(x)$")
        ax.set_title(fr"$f(x) = x^{{{n_dii}}} - {M_dii} \ $, in selected interval [{a_dii},{b_dii}]")
        ax.grid(ls='--', lw = 0.75, alpha = 0.6, c='grey')
        ax.legend()
        plt.tight_layout()
        fig_name = f'fx-nth-root-in-sol-interval-{a_dii}-to-{b_dii}-n-eq-{n_dii}-M-eq-{M_dii}'
        plt.savefig(f"plots/{fig_name}.png")
        plt.savefig(f"plots/{fig_name}.pdf")
        file_write(output_file, [f"\nPlot saved as {fig_name}.png and {fig_name}.pdf in 'plots' directory\n\n"])
        plt.show()



#creating the plots directory to store the plots
os.makedirs("plots", exist_ok=True)

#creating the output file
output_file = open("output.txt","w")
file_write(output_file, ["2024PHY1005_Jaskirat-Singh_Computational-Lab-Assignment-4_Terminal-Output\n\n\n\n"])



#menu approach to find multiple roots for any function (tan(x)-x or nth root) in same program
menu_dict = {"Choice":[1, 2, 3],
        "Task":["Finding roots for f(x) = tan(x) - x", "Finding nth root of number M", "Exit"]}
menu_table = tabulate(menu_dict, headers='keys', tablefmt='grid')


while True:
    file_write(output_file, ["\n\n===========Menu===========\n\n",
                             menu_table+"\n\n"])
    choice = int(input("Enter your choice from above menu: "))
    file_write(output_file, [f"\n\nChoice made: {choice}\n\n"])
    if choice in [1,2,3]:
        if choice==1:
            file_write(output_file, ["\n\n---------Root finding for f(x) = tan(x) - x---------\n\n"])
            tan_func_root_find()
        elif choice==2:
            file_write(output_file, ["\n\n---------Finding nth root of a number M---------\n\n"])
            nth_root_func_root_find()
        elif choice==3:
            file_write(output_file, ["\n\nExiting\n\n"])
            break
    else:
        file_write(output_file, ["\n\nInvalid Choice\n\n"])


file_write(output_file, ["\n\n\n\nEnd of Program"])

#closing the file
output_file.close()
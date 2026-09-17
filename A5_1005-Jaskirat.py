import numpy as np
from phyFuncs import *
import os #to create a separate directory to save plots
from tabulate import tabulate
import matplotlib.pyplot as plt
import time #to calculate time utilized by the code to execute
from scipy.constants import hbar, m_e, m_p

#function for saving terminal output
def file_write(output_file,
               output_list: list) -> None:
    """
    Writes output into a file and prints into the terminal

    Inputs:
    output_file: a file object to store terminal output;
    output_list: a list containing the lines of terminal output
    """
    output_file.writelines(output_list)
    for line in output_list: print(line, end='')

#dimensionless TISE sys of linear ODEs
def y1_prime(x: np.ndarray | int | float,
             y: list,
             *args, **kwargs) -> float:
    """
    Returns value of y1 for the TISE with potential zero inside infinite well

    Inputs:
    x: value of independent variable;
    y: list containing the two variables obtained while converting 2nd order TISE to system of linear differential equations

    Outputs:
    ans: value of y1 at given step
    """
    y1, y2 = y
    ans = y2
    return(ans)

#dimensionless TISE sys of linear ODEs
def y12_prime(x: np.ndarray | int | float,
             y: list,
             epsilon: int | float = 1,
             *args, **kwargs) -> float:
    """
    Returns value of y2 for the TISE with potential zero inside infinite well

    Inputs:
    x: value of independent variable;
    y: list containing the two variables obtained while converting 2nd order TISE to system of linear differential equations

    Outputs:
    ans: value of y2 at given step
    """
    y1, y2 = y
    ans = -epsilon*y1
    return(ans)

#dimensionless TISE sys of linear ODEs when potential inside well is of form K*x^2
def y22_prime(x: np.ndarray | int | float,
             y: list,
             epsilon: int | float = 1,
             K: int | float = 0,
             *args, **kwargs) -> float:
    """
    Returns value of y2 for the TISE with potential as k*x^2 inside infinite well

    Inputs:
    x: value of independent variable;
    y: list containing the two variables obtained while converting 2nd order TISE to system of linear differential equations

    Outputs:
    ans: value of y2 at given step
    """
    y1, y2 = y
    ans = -(epsilon - (K*(x**2)))*y1
    return(ans)


#initializing the time at start
time_start = time.perf_counter()

#creating the plots directory to store the plots
os.makedirs("plots", exist_ok=True)

#creating the output file
output_file = open("output.txt","w")
file_write(output_file, ["2024PHY1005_Jaskirat-Singh_Computational-Lab-Assignment-5_Terminal-Output\n\n\n\n"])



#initializing parameters
epsilon = np.linspace(0, 1100, 100) #100 values of epsilon
ll = -0.5; ul = 0.5; h = 0.0001; n = int((ul-ll)/h)

func = [y1_prime, y12_prime]
alpha = [0, 1] #initial values of y1 and y2 
u_at_right_boundary = 0

#calculating residue at each energy
y = [my_RK4(func=func, a = ll, b = ul, alpha = alpha, n = n, epsilon = eps_i) for eps_i in epsilon]
x_vals = y[0][0]
residues = []

for indx in range(len(epsilon)):
    residues.append(y[indx][1][0][-1] - u_at_right_boundary)


#plotting residues with energy
fig, ax = plt.subplots(1, 1, figsize = (10, 6), num = "Residue Plot")

ax.plot(epsilon, residues, '.', markersize = 8)
ax.axhline(0, c = 'black', lw = 0.75)
ax.set_title(fr"Residue v/s $\epsilon$, h = {h}")
ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel(r'$R(\epsilon)$')
ax.grid(c = 'grey', ls = '--', lw = 0.75, alpha = 0.6)




#printing the indices at which the sign changes to determine the interval
residual_arr = np.array(residues)
residual_arr[residual_arr==0] = 1
sign_change = np.where(np.diff(np.sign(residual_arr)) != 0)[0]
sign_change_energies = [epsilon[indx] for indx in sign_change]

dict_sign_change = {"Indices":sign_change,
                    "Energy Value": sign_change_energies}
table = tabulate(dict_sign_change, headers = 'keys', tablefmt='grid')
file_write(output_file, ["The Energy values where the sign of residue changes are as below\n",
                         table+"\n\n"])


#applying shooting method for all possible solutions in the energy interval
energy_sols = []

for indx in range(len(sign_change)): #runs for the number of times I get roots in the residue plot
    sign_change_indx = sign_change[indx]
    x = [epsilon[sign_change_indx], epsilon[sign_change_indx+1]]
    y = [residues[sign_change_indx], residues[sign_change_indx+1]]

    y = shooting(func=func, a = ll, b = ul, alpha = alpha, n = n, u_at_right_boundary = u_at_right_boundary, x=x, y=y)

    energy_sols.append(y)

#defining the states (first 5 bound) to calculate exact energies and show on plot to compare with numerical ones graphically
n_states = 5
num_states_to_plot = n_states if n_states<=len(sign_change) else len(sign_change)
if n_states>len(sign_change): file_write(output_file, [f"\n\n--> Can not obtain energies for first {n_states} bound states since solution to only the first {len(sign_change)} bound states exist in the energy interval chosen\n\n"])

states = list(range(1, len(sign_change) + 1))
exact_energies = [(n*np.pi)**2 for n in states]

dict_energy_sols = {"State": states[:num_states_to_plot],
                    "Energy Num Sol":energy_sols[:num_states_to_plot],
                    "Exact Energy":exact_energies[:num_states_to_plot],
                    "Relative Error":[f"{(e_i - e_exact_i)*100/e_exact_i:.2f} %" for e_i,e_exact_i in zip(energy_sols[:num_states_to_plot], exact_energies[:num_states_to_plot])]}
table = tabulate(dict_energy_sols, headers = 'keys', tablefmt = 'grid')
file_write(output_file, [f"\n\nEnergy Solutions obtained numerically for first {num_states_to_plot} states are as below\n\n",
                         table+"\n\n"])

#including the exact energies calculated above in the plot of residue
ax.scatter(exact_energies, np.asarray(exact_energies)*0, marker='*', c='orange', label = "Exact Energies")
ax.legend()

fig_name = "Residue_plot_using_RK4"
plt.tight_layout()
plt.savefig(f"plots/{fig_name}.png")
plt.savefig(f"plots/{fig_name}.pdf")

file_write(output_file, [f"\n\nPlots saved as {fig_name}.png and {fig_name}.pdf in 'plots' directory\n\n"])


#calculating numerical normalized wavefunctions from the energy solutions
y_num_sol = [my_RK4(func=func, a = ll, b = ul, alpha = alpha, n = n, epsilon = eps_i)[1][0] for eps_i in energy_sols[:num_states_to_plot]]
y_num_sol_arr = np.asarray(y_num_sol)
A_num_list = [1/np.sqrt(trapezoidal(func = y_num_sol_arr[indx]**2, a = ll, b = ul)) for indx in range(len(energy_sols[:num_states_to_plot]))]
y_num_normalized_sol = [A_i * y_i for A_i, y_i in zip(A_num_list, y_num_sol_arr)]

#calculating analytical normalized wavefunctions from the energy solutions
y_ana_sol = [my_RK4(func=func, a = ll, b = ul, alpha = alpha, n = n, epsilon = eps_i)[1][0] for eps_i in exact_energies[:num_states_to_plot]]
y_ana_sol_arr = np.asarray(y_ana_sol)
A_ana_list = [1/np.sqrt(trapezoidal(func = y_ana_sol_arr[indx]**2, a = ll, b = ul)) for indx in range(len(exact_energies[:num_states_to_plot]))]
y_ana_normalized_sol = [A_i * y_i for A_i, y_i in zip(A_ana_list, y_ana_sol_arr)]


#plotting the normalized wavefuctions
fig1, ax1 = plt.subplots(num_states_to_plot, 1, figsize=(10, num_states_to_plot*3), sharex = True, num = "Normalized Wavefunctions")
#plotting the probability densities
fig2, ax2 = plt.subplots(num_states_to_plot, 1, figsize=(10, num_states_to_plot*3), sharex = True, num = "Probability Densities")

for indx in range(len(y_num_normalized_sol)):
    new_indx = (len(y_num_normalized_sol)-1)-indx
    #normalized wave functions
    ax1[new_indx].plot(x_vals, y_num_normalized_sol[indx], '.', markersize = 4, label = fr"$n = {indx+1}$")
    ax1[new_indx].plot(x_vals, y_ana_normalized_sol[indx])
    ax1[new_indx].axhline(0, c = 'black', lw = 0.75)
    ax1[new_indx].axvline(0, c = 'black', lw = 0.75)
    ax1[new_indx].set_ylabel(r"$u(\xi)$")
    ax1[new_indx].grid(lw = 0.75, alpha = 0.6, ls = '--', c = 'grey')
    ax1[new_indx].legend()
    #probability densities
    ax2[new_indx].plot(x_vals, y_num_normalized_sol[indx]**2, '.', markersize = 4, label = fr"$n = {indx+1}$")
    ax2[new_indx].plot(x_vals, y_ana_normalized_sol[indx]**2)
    ax2[new_indx].axhline(0, c = 'black', lw = 0.75)
    ax2[new_indx].axvline(0, c = 'black', lw = 0.75)
    ax2[new_indx].set_ylabel(r"$u(\xi)$")
    ax2[new_indx].grid(lw = 0.75, alpha = 0.6, ls = '--', c = 'grey')
    ax2[new_indx].legend()

#normalized wave functions
ax1[-1].set_xlabel(r"$\xi$")
ax1[0].set_title(fr"Plot of all normalized solutions in first {indx+1} bounded states $n$"
                "\nNumerical as discrete points, analytical as continuous curve")
fig_name1 = f"Normalized-sols-{indx+1}-bound-states"
fig1.tight_layout()
fig1.savefig(f"plots/{fig_name1}.png")
fig1.savefig(f"plots/{fig_name1}.pdf")

#probability densities
ax2[-1].set_xlabel(r"$\xi$")
ax2[0].set_title(fr"Plot of probability densities in first {indx+1} bounded states $n$"
                "\nNumerical as discrete points, analytical as continuous curve")
fig_name2 = f"Prob-densities-for-{indx+1}-bound-states"
fig2.tight_layout()
fig2.savefig(f"plots/{fig_name2}.png")
fig2.savefig(f"plots/{fig_name2}.pdf")

#normalized wave functions
file_write(output_file, [f"\n\nPlots saved as {fig_name1}.png and {fig_name1}.pdf in 'plots' directory\n\n"])

#probability densities
file_write(output_file, [f"\n\nPlots saved as {fig_name2}.png and {fig_name2}.pdf in 'plots' directory\n\n"])


#plotting calculated energies as a function of the square of the state
n_sq_arr = np.array([n_i**2 for n_i in states])
energy_sols_arr = np.asarray(energy_sols)

fig, ax = plt.subplots(1, 1, figsize = (6, 6), num = "Numerically calculated dimensionless energy vs state squared")
ax.plot(n_sq_arr, energy_sols_arr, '.', markersize = 8, label = "Data points")

params = lsf(x = n_sq_arr, y = energy_sols_arr)
if params!=None:
    mean, sigma, m, c = params
    ax.plot(n_sq_arr, ((m*n_sq_arr) + c), ls = '--', c = 'r', label = "Best Fit")
    # ax.errorbar(n_sq_arr, ((m*n_sq_arr) + c), fmt = '.', yerr = sigma, capsize = 5, c = 'red')

ax.axhline(0, lw = 0.75, c = 'black')
ax.axvline(0, lw = 0.75, c = 'black')
ax.set_title(r"$e_{n}$ v/s $n^{2}$")
ax.set_xlabel(r"$n^{2}$")
ax.set_ylabel(r"$e_{n}$")
ax.grid(ls = '--', lw = 0.75, c = 'grey', alpha = 0.6)
fig.tight_layout()
fig_name = "en-vs-n_sq"
fig.savefig(f"plots/{fig_name}.png")
fig.savefig(f"plots/{fig_name}.pdf")
file_write(output_file, [f"\n\nPlots saved as {fig_name}.png and {fig_name}.pdf in 'plots' directory\n\n"])

table = tabulate({" ": ["Slope"],
                 "Numerical":[f"{m:.3f}"],
                 "Exact":[f"{np.pi**2:.3f}"],
                 "Relative Error":[f"{(m-np.pi**2)*100/np.pi**2:.2f} %"]},
                 headers = 'keys', tablefmt = 'grid')
file_write(output_file, ["\n\nAnalysis of the slope for the Best Fit Line\n",
                         table+'\n\n'])



#converting the dimensionless energies into eV
L = 5*1e-10  #5 angstrom = total length of the potential well
factor = hbar**2/(2*m_e*(L**2))

#the energies in units eV
exact_energies_eV = [f"{e_i*factor:.6g}" for e_i in exact_energies]
energy_sols_eV = [f"{e_i*factor:.6g}" for e_i in energy_sols]

dict_dimension_energies = {"State":states,
                           "Numerical (eV)":energy_sols_eV,
                           "Exact (eV)":exact_energies_eV,
                           "Relative Error": [f"{(float(e_ana_i)-float(e_exact_i))*100/float(e_exact_i):2f} %" for e_ana_i,e_exact_i in zip(energy_sols_eV, exact_energies_eV)]}
table = tabulate(dict_dimension_energies, headers = 'keys', tablefmt = 'grid')
file_write(output_file, [f"\n\nThe Energies for various states of an electron in Potential well of width {L} angstrom are as follows\n",
                         table+'\n\n'])


#finding probability of particle in this well of width 5 angstrom from -3L/8 to L/4
integrate_ll = -3/8
integrate_ul = 1/4

#initializing parameters
widths = [5*1e-10, 10*1e-10, 5*1e-15] #electron in 5 ang, 10 ang and proton in 5 fermimeters
masses = [m_e, m_e, m_p]  #electron, electron, proton
names_list = ["electron", "electron", "proton"]

states_integrate = [0,1]

mask = (x_vals>=integrate_ll) & (x_vals<=integrate_ul)
y_num_normalized_sol_arr = [np.asarray(y_num_normalized_sol[state]) for state in states_integrate]  #0 and 1 correspond to the ground and 1st excited states respectively
y_ana_normalized_sol_arr = [np.asarray(y_ana_normalized_sol[state]) for state in states_integrate]  #0 and 1 correspond to the ground and 1st excited states respectively


for L, M, indx in zip(widths, masses, range(len(names_list))):
    numerical, analytic, rel_err = [], [], []
    energy_factor = hbar**2/(2*M*(L**2))


    for y_ana_i, y_num_i in zip(y_ana_normalized_sol_arr, y_num_normalized_sol_arr): #ground state at 0 indx and 1st excited at indx 1
        #integrate the functions for all the systems
        integrate_func_num = (y_num_i[mask])**2
        integrate_func_ana = (y_ana_i[mask])**2
        integrate_sum_num = simpson_1_3(func = integrate_func_num/L, a = integrate_ll*L, b = integrate_ul*L)
        integrate_sum_ana = simpson_1_3(func = integrate_func_ana/L, a = integrate_ll*L, b = integrate_ul*L)
        numerical.append(integrate_sum_num)
        analytic.append(integrate_sum_ana)
        rel_err.append((integrate_sum_num-integrate_sum_ana)*100/integrate_sum_ana)

    dict_integrate = {"State":["Ground", "First Excited"],
                      "Numerical":[f"{num*100:.2f} %" for num in numerical],
                      "Analytical":[f"{ana*100:.2f} %" for ana in analytic],
                      "Relative Error":[f"{re:.2f} %" for re in rel_err]}
    table = tabulate(dict_integrate, headers = 'keys', tablefmt = 'grid')
    file_write(output_file, [f"\n\nProbabilities of finding {names_list[indx]} from [-3L/8, L/4] in a Potential Well of width {L:.1g} meters when in Ground and First Excited States\n",
                             table+'\n\n'])


#================================================================================

"""Solving the Schrodinger Equation for the Infinite Well with non-zero potential inside"""
file_write(output_file, ["\n\n\n==========Solving the Schrodinger Equation for the Infinite Well with quadratic potential inside==========\n\n\n"])
# K = 500  #strength of the quadratic deformation potential

kappa = [500, 100, 50, 10, 5, 1, 0]

for K in kappa:

    file_write(output_file, [f"\n\n\n======For K={K}======\n\n\n"])

    V_x = K*(x_vals**2)

    fig, ax = plt.subplots(1, 1, figsize = (6, 6), num = f"Quadratic Potential Visualization with K={K}")
    ax.plot(x_vals, V_x)
    ax.axhline(0, lw = 0.75, c = 'black')
    ax.axvline(0, lw = 0.75, c = 'black')
    ax.set_title(fr"Potential Visualization of the Well, $V(\xi)=\kappa x^2$, $\kappa={K}$")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$V(\xi)$")
    ax.grid(c='grey', ls = '--', lw = 0.75, alpha = 0.6)
    fig.tight_layout()
    fig_name = f"Potential-Visualization_Infinite-Well_V-K-x_sq-K-equals-{K}"
    fig.savefig(f"plots/{fig_name}.png")
    fig.savefig(f"plots/{fig_name}.pdf")
    file_write(output_file, [f"\n\nPlots saved as {fig_name}.png and {fig_name}.pdf in the 'plots' directory\n\n"])


    func = [y1_prime, y22_prime]
    h = 0.0001; n = int((ul-ll)/h)
    epsilon = np.linspace(0, 1100, 100) #100 values of epsilon

    #calculating residue at each energy
    y = [my_RK4(func=func, a = ll, b = ul, alpha = alpha, n = n, epsilon = eps_i, K = K) for eps_i in epsilon]
    x_vals = y[0][0]
    residues = []

    for indx in range(len(epsilon)):
        residues.append(y[indx][1][0][-1] - u_at_right_boundary)


    #plotting residues with energy
    fig, ax = plt.subplots(1, 1, figsize = (10, 6), num = f"Residue Plot for Quadratic Potential with K={K}")

    ax.plot(epsilon, residues, '.', markersize = 8)
    ax.axhline(0, c = 'black', lw = 0.75)
    ax.set_title(fr"Residue v/s $\epsilon$, $V(x)=Kx^2$, K={K}, h={h}")
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$R(\epsilon)$')
    ax.grid(c = 'grey', ls = '--', lw = 0.75, alpha = 0.6)


    #printing the indices at which the sign changes to determine the interval
    residual_arr = np.array(residues)
    # residual_arr[residual_arr==0] = 1
    sign_change = np.where(np.diff(np.sign(residual_arr)) != 0)[0]
    sign_change_energies = [epsilon[indx] for indx in sign_change]

    dict_sign_change = {"Indices":sign_change,
                        "Energy Value": sign_change_energies}
    table = tabulate(dict_sign_change, headers = 'keys', tablefmt='grid')
    file_write(output_file, [f"\nThe Energy values where the sign of residue changes with K={K} are as below\n",
                            table+"\n\n"])


    #applying shooting method for all possible solutions in the energy interval
    energy_sols = []

    for indx in range(len(sign_change)): #runs for the number of times I get roots in the residue plot
        sign_change_indx = sign_change[indx]
        x = [epsilon[sign_change_indx], epsilon[sign_change_indx+1]]
        y = [residues[sign_change_indx], residues[sign_change_indx+1]]

        y = shooting(func=func, a = ll, b = ul, alpha = alpha, n = n, u_at_right_boundary = u_at_right_boundary, x=x, y=y, K = K)

        energy_sols.append(y)

    #defining the states (first 5 bound) to calculate exact energies and show on plot to compare with numerical ones graphically
    n_states = 5
    num_states_to_plot = n_states if n_states<=len(sign_change) else len(sign_change)
    if n_states>len(sign_change): file_write(output_file, [f"\n\n--> Can not obtain energies with K={K} for first {n_states} bound states since solution to only the first {len(sign_change)} bound states exist in the energy interval chosen\n\n"])

    states = list(range(1, len(sign_change) + 1))
    exact_energies = [(n*np.pi)**2 for n in states]

    dict_energy_sols = {"State": states[:num_states_to_plot],
                        "Energy Num Sol":energy_sols[:num_states_to_plot],
                        "Exact Energy ((n pi)^2)":exact_energies[:num_states_to_plot],
                        "Relative Error":[f"{(e_i - e_exact_i)*100/e_exact_i:.2f} %" for e_i,e_exact_i in zip(energy_sols[:num_states_to_plot], exact_energies[:num_states_to_plot])]}
    table = tabulate(dict_energy_sols, headers = 'keys', tablefmt = 'grid')
    file_write(output_file, [f"\n\nEnergy Solutions obtained numerically for first {num_states_to_plot} states with K={K} are as below\n\n",
                            table+"\n\n"])

    #including the exact energies calculated above in the plot of residue
    ax.scatter(exact_energies, np.asarray(exact_energies)*0, marker='*', c='orange', label = "Exact Energies")
    ax.legend()

    fig_name = f"Residue_plot_using_RK4_V-K-x_sq-K-equals-{K}"
    plt.tight_layout()
    plt.savefig(f"plots/{fig_name}.png")
    plt.savefig(f"plots/{fig_name}.pdf")

    file_write(output_file, [f"\n\nPlots saved as {fig_name}.png and {fig_name}.pdf in 'plots' directory\n\n"])



    #calculating numerical normalized wavefunctions from the energy solutions
    y_num_sol = [my_RK4(func=func, a = ll, b = ul, alpha = alpha, n = n, epsilon = eps_i, K=K)[1][0] for eps_i in energy_sols[:num_states_to_plot]]
    y_num_sol_arr = np.asarray(y_num_sol)
    A_num_list = [1/np.sqrt(trapezoidal(func = y_num_sol_arr[indx]**2, a = ll, b = ul)) for indx in range(len(energy_sols[:num_states_to_plot]))]
    y_num_normalized_sol = [A_i * y_i for A_i, y_i in zip(A_num_list, y_num_sol_arr)]

    #calculating analytical normalized wavefunctions from the energy solutions
    y_ana_sol = [my_RK4(func=func, a = ll, b = ul, alpha = alpha, n = n, epsilon = eps_i, K=K)[1][0] for eps_i in exact_energies[:num_states_to_plot]]
    y_ana_sol_arr = np.asarray(y_ana_sol)
    A_ana_list = [1/np.sqrt(trapezoidal(func = y_ana_sol_arr[indx]**2, a = ll, b = ul)) for indx in range(len(exact_energies[:num_states_to_plot]))]
    y_ana_normalized_sol = [A_i * y_i for A_i, y_i in zip(A_ana_list, y_ana_sol_arr)]


    #plotting the normalized wavefuctions
    fig, ax = plt.subplots(num_states_to_plot, 1, figsize=(10, num_states_to_plot*3), sharex = True, num = f"Normalized Solutions for Quadratic Potential with K={K}")

    for indx in range(len(y_num_normalized_sol)):
        new_indx = (len(y_num_normalized_sol)-1)-indx
        
        ax[new_indx].plot(x_vals, y_num_normalized_sol[indx], '.', markersize = 4, label = fr"$n = {indx+1}$")
        ax[new_indx].plot(x_vals, y_ana_normalized_sol[indx])
        ax[new_indx].axhline(0, c = 'black', lw = 0.75)
        ax[new_indx].axvline(0, c = 'black', lw = 0.75)
        ax[new_indx].set_ylabel(r"$u(\xi)$")
        ax[new_indx].grid(lw = 0.75, alpha = 0.6, ls = '--', c = 'grey')
        ax[new_indx].legend()

    #normalized wave functions
    ax[-1].set_xlabel(r"$\xi$")
    ax[0].set_title(fr"Plot of all normalized solutions in first {indx+1} bounded states $n$"
                    "\nNumerical as discrete points, analytical as continuous curve\n"
                    r"$V(\xi)=\kappa \xi^2$"
                    fr", $\kappa={K}$")
    fig_name = f"Normalized-sols-{indx+1}-bound-states_V-K-x_sq-K-equals-{K}"
    fig.tight_layout()
    fig.savefig(f"plots/{fig_name}.png")
    fig.savefig(f"plots/{fig_name}.pdf")

    file_write(output_file, [f"\n\nPlots saved as {fig_name}.png and {fig_name}.pdf in 'plots' directory\n\n"])




time_end = time.perf_counter()

file_write(output_file, [f"\n\n\n\nEnd of Program --- Time taken: {time_end - time_start:.2f} sec"])

#closing the file
output_file.close()

plt.show()
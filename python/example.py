from Gutzwiller import GutzwillerSolver
from Explorer2D import Explorer2D
import h5py
import numpy as np

np.set_printoptions(precision=4, linewidth=200, suppress=True)


def get_dos(filename=None):
    if filename == None:
        W = 1
        n_points = 1000
        de = W / n_points
        energies = [-W / 2 + (n + 0.5) * de for n in range(n_points)]
        weights = [1 / W * de] * n_points
        return energies, weights
    else:
        dos = np.load('dos/' + filename)
        return dos[0].tolist(), dos[1].tolist()

def hf_lambda_and_eta():
    lam = [1] * 12 + [0] * 6
    eta = [0] * 5
    return lam, eta

solver = GutzwillerSolver()

# Set the accuracy and other parameters of the solver

solver.set_solver_params(ga_error=1e-5, ga_x = 0.5, rho_error=1e-7, rho_x=0.5, lambda_error=1e-7, com_error=1e-7, n_error=0.0001)

# Set the density of states from file or a constant one by default

energies, weights = get_dos()
solver.set_dos(energies, weights)

# Set the model parameters

n = 3.25
U = 1.0
J = 0.5
e_f = -0.5
V = 0.5
T = 0.00001

params = dict(n=n, e_f=e_f, V=V, U=U, U_p=(U - 2 * J), J=J, J_c=J, T=T)

for name, value in params.items():
    solver.set_model_param(value, name)

# Choose a phase

phase = 'FMSC'

# Set the initial values for rho, lambda eta and mu

rho_init =  [ 0.8,  0.2 , 0. , 0.05, -0.4, -0.3,  0,  -0.05]
lambda_init, eta_init =  hf_lambda_and_eta()
mu_init = -0.35

solver.set_rho(rho_init)
solver.set_model_param(mu_init, 'mu')
solver.set_lambda(lambda_init)
solver.set_eta(eta_init)

# First, find Hartree-Fock solution

solver.set_phase(phase, hartree_fock=True)

solver.solve(mode='rho')

print('Rho:    ', repr(np.array(solver.get_rho())))
print('Mu:     ', round(solver.get_model_param('mu'), 4))
print('Lambda: ', repr(np.array(solver.get_lambda())))
print('Eta:    ', repr(np.array(solver.get_eta())))
print('Energy:    ', round(solver.get_energy('full'), 4))

# Then, find the solution of Gutzwiller approximation

solver.set_phase(phase, hartree_fock=False)

solver.solve(mode='full')

print('Rho:    ', repr(np.array(solver.get_rho())))
print('Mu:     ', round(solver.get_model_param('mu'), 4))
print('Lambda: ', repr(np.array(solver.get_lambda())))
print('Eta:    ', repr(np.array(solver.get_eta())))
print('Energy:    ', round(solver.get_energy('full'), 4))


# Make sure it corresponds to energy minimum - all the minors listed below should be negative

print(solver.lambda_hessian_minors())

# Write the result to the file

meta = 'funny_name'

with h5py.File('DATA/' + Explorer2D.filename(phase, params, [], meta=meta, hf=False), 'w') as file:
    Explorer2D.save_solution(solver, file)
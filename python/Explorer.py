import numpy as np
import h5py
from Gutzwiller import GutzwillerSolver


class Explorer:
    def __init__(self, solver):
        self.solver = solver
        self.set_phase('FMSC')
        self.varying_parameter = 'V'
        self.range = [0, 5]
        self.step = 0.1
        self.constant_parameters = ['U', 'J', 'n', 'T']
        self.additional_text = ''

    def set_phase(self, phase_name, hartree_fock = False):
        self.phase = phase_name
        self.hf = hartree_fock
        self.solver.set_phase(phase_name, hartree_fock)

    def exploration(self, call_on_solver, additional_text='', append=True, precise=False, rho_iterations=False):
        if self.range[0] <= self.range[1]:
            # ascending order
            grid = np.arange(self.range[0], self.range[1] + 0.5 * self.step, self.step)
        else:
            # descending order
            grid = np.arange(self.range[1], self.range[0] + 0.5 * self.step, self.step)[::-1]


        file = h5py.File('DATA/' + self.this_filename(additional_text), ('a' if append == True else 'w'))
        stuck_flag = 0
        for value in grid:
            # Update the parameter
            self.solver.set_model_param(value, self.varying_parameter)
            print('Starting optimization...', flush=True)
            print(self.varying_parameter + ': ' + str(value), flush=True)
            print(' ', flush=True)
            # Apply constraints, etc
            call_on_solver(self.solver)
            # Calculate the parameters

            #self.solver.solve(mode='full')
            if(not self.hf):
                if self.solver.solve(mode=('combined' if precise else 'full')) != 0:
                    print('Trying again.', flush=True)
                    self.solver.solve(mode = 'iterations')
                    if self.solver.solve(mode = 'combined') != 0:
                        print('We are stuck. Sorry.', flush=True)
                        break
            else:
                if rho_iterations:
                    self.solver.solve_fixing_n(n=self.solver.get_model_param('n'),
                                               mu_bounds=[self.solver.get_model_param('mu') - 0.01,
                                                          self.solver.get_model_param('mu') + 0.01], mode='rho_iterations')
                else:
                    if self.solver.solve(mode='rho') != 0:
                        self.solver.solve(mode='rho_iterations')
                        self.solver.solve_fixing_n(n=self.solver.get_model_param('n'),
                                                   mu_bounds=[self.solver.get_model_param('mu') - 0.0005,
                                                              self.solver.get_model_param('mu') + 0.0005],
                                                   mode='rho_iterations')
                        if self.solver.solve(mode='rho') != 0:
                            print('We are stuck. Sorry.', flush=True)
                            break


            print(' ', flush=True)
            # Write solution to file
            self.this_write_to_file(file, value)
        file.close()

    def this_filename(self, additional_text=''):
        const_params_values = []
        for param in self.constant_parameters:
            const_params_values.append(self.solver.get_model_param(param))
        return Explorer.filename(self.phase, self.constant_parameters, const_params_values, self.varying_parameter, additional_text)

    @staticmethod
    def write_t_file(solver, f, var_param_value):
        sol = f.require_group(str(var_param_value))
        sol['rho'] = solver.get_rho()
        sol['lambda'] = solver.get_lambda()
        sol['eta'] = solver.get_eta()
        sol['model'] = ()
        for param in ('T', 'mu', 'e_f', 'V', 'U', 'U_p', 'J', 'J_c', 'n'):
            sol['model'].attrs[param] = solver.get_model_param(param)
        sol['observable'] = ()
        for observable in ('n_f_up', 'n_f_do', 'A_f_up', 'A_f_do', 'v_up', 'v_do', 'A_cf_up', 'A_cf_do'):
            sol['observable'].attrs[observable] = solver.get_correlated_observable(observable)
        for observable in ('n_c_up', 'n_c_do', 'A_c_up', 'A_c_do'):
            sol['observable'].attrs[observable] = solver.get_uncorrelated_observable(observable)
        sol['uncorrelated_observable'] = ()
        for observable in ('n_f_up', 'n_f_do', 'A_f_up', 'A_f_do', 'v_up', 'v_do', 'A_cf_up', 'A_cf_do',
                           'n_c_up', 'n_c_do', 'A_c_up', 'A_c_do'):
            sol['uncorrelated_observable'].attrs[observable] = solver.get_uncorrelated_observable(observable)
        sol['energy'] = ()
        for energy_part in ('full', 'f', 'fc', 'c'):
            sol['energy'].attrs[energy_part] = solver.get_energy(energy_part)
        f.flush()

    def this_write_to_file(self, f, var_param_value):
        self.write_t_file(self.solver, f, var_param_value)


    @staticmethod
    def filename(phase, const_params, const_params_values, var_param, additional_text='', suffix=True):
        phase_string = ''
        for p in phase:
            phase_string += p
        param_string = ''
        for param, value in zip(const_params, const_params_values):
            param_string += param + '{:.6f}'.format(value) + '_'
        param_string += var_param
        return  'GA_' + phase_string + '_' + param_string + additional_text + ('.hdf5' if suffix else '')

    @staticmethod
    def get_observable(param_name, sol):
        if param_name == 'energy':
            return sol['energy'].attrs['full']
        elif param_name == 'mu':
            return sol['model'].attrs['mu']
        elif param_name == 'n':
            return sol['model'].attrs['n']
        obs = sol['observable'].attrs
        m_f = 2 * (obs['n_f_up'] - obs['n_f_do']) / 2.
        I = sol['model'].attrs['J'] - sol['model'].attrs['U_p']
        if param_name == 'm_f':
            return abs(m_f)
        elif param_name == 'm_c':
            return 2 * (1 if m_f >= 0 else -1) * (obs['n_c_up'] - obs['n_c_do']) / 2.
        elif param_name == 'm':
            return Explorer.get_observable('m_f', sol) + Explorer.get_observable('m_c', sol)
        elif param_name == 'n_f':
            return obs['n_f_up'] + obs['n_f_do']
        elif param_name == 'n_c':
            return obs['n_c_up'] + obs['n_c_do']
        elif param_name == 'Delta_f_up':
            return I * abs(obs['A_f_up']) if m_f >= 0 else I * abs(obs['A_f_do'])
        elif param_name == 'Delta_f_do':
            return I * abs(obs['A_f_up']) if m_f < 0 else I * abs(obs['A_f_do'])
        elif param_name == 'Delta_c_up':
            return I * abs(obs['A_c_up']) if m_f >= 0 else I * abs(obs['A_c_do'])
        elif param_name == 'Delta_c_do':
            return I * abs(obs['A_c_up']) if m_f < 0 else I * abs(obs['A_c_do'])
        elif param_name == 'Delta_cf_up':
            return I * abs(obs['A_cf_up']) if m_f >= 0 else I * abs(obs['A_cf_do'])
        elif param_name == 'Delta_cf_do':
            return I * abs(obs['A_cf_up']) if m_f < 0 else I * abs(obs['A_cf_do'])

        elif param_name == 'A_f_up':
            return abs(obs['A_f_up']) if m_f >= 0 else abs(obs['A_f_do'])
        elif param_name == 'A_f_do':
            return abs(obs['A_f_up']) if m_f < 0 else abs(obs['A_f_do'])
        elif param_name == 'A_c_up':
            return abs(obs['A_c_up']) if m_f >= 0 else abs(obs['A_c_do'])
        elif param_name == 'A_c_do':
            return abs(obs['A_c_up']) if m_f < 0 else abs(obs['A_c_do'])
        elif param_name == 'A_cf_up':
            return abs(obs['A_cf_up']) if m_f >= 0 else abs(obs['A_cf_do'])
        elif param_name == 'A_cf_do':
            return abs(obs['A_cf_up']) if m_f < 0 else abs(obs['A_cf_do'])
        else:
            return obs[param_name]


    @staticmethod
    def load_params_from_file(solver, phase, const_params, const_params_values, var_param, var_param_value, additional_text=''):
        filename = Explorer.filename(phase, const_params, const_params_values, var_param, additional_text)
        print(filename)
        f = h5py.File('DATA/' + filename)
        sol = f[str(var_param_value)]
        solver.set_rho(sol['rho'][:].tolist())
        solver.set_lambda(sol['lambda'][:].tolist())
        solver.set_eta(sol['eta'][:].tolist())
        for param in ('T', 'mu', 'e_f', 'V', 'U', 'U_p', 'J', 'J_c', 'n'):
            solver.set_model_param(sol['model'].attrs[param], param)

    @staticmethod
    def load_params_from_solutions_file(solver, phase, const_params, const_params_values, var_param, var_param_value, energy):
        f = h5py.File('SOL/' + Explorer.filename(phase, const_params, const_params_values, var_param,
                                    additional_text=('{:.6f}'.format(var_param_value)) + '_unique_solutions'), 'r')
        sol = f['{0:0{width}}'.format(energy, width=4)]
        solver.set_rho(sol['rho'][:].tolist())
        solver.set_lambda(sol['lambda'][:].tolist())
        solver.set_eta(sol['eta'][:].tolist())
        solver.set_model_param(sol['mu'][0], 'mu')


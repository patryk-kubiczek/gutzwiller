import numpy as np
from copy import deepcopy
import h5py

def float_to_str(number):
    return '{:.6f}'.format(number)

order = ('n', 'U', 'J', 'e_f', 'V', 'U_p', 'T', 'J_c')


class Explorer2D:
    def __init__(self, solver):
        self.solver = solver
        self.set_mode('FMSC', hartree_fock=False)
        self.x_param = 'V'
        self.y_param = 'e_f'
        self.set_const_params({p: solver.get_model_param(p) for p in ('n', 'U', 'J')})
        self.meta = ''

    def set_mode(self, phase_name, hartree_fock = False):
        self.phase = phase_name
        self.hf = hartree_fock
        self.solver.set_phase(phase_name, hartree_fock)


    def set_const_params(self, const_params):
        self.const_params = const_params
        for param, value in const_params.items():
            self.solver.set_model_param(value, param)

    def append_col(self, x_value, meta=None):
        const_params = deepcopy(self.const_params)
        const_params[self.x_param] = x_value
        filename = 'DATA/' + self.filename(self.phase, const_params, (self.y_param, ), meta, self.hf)
        source_file = h5py.File(filename)
        destination_file = h5py.File('DATA2D/' + self.this_filename())
        print('Appending data from ' + filename)
        for y, sol in source_file.items():
            dest = destination_file.require_group(float_to_str(float(y)))
            if float_to_str(x_value) in dest.keys():
                del (dest[float_to_str(x_value)])
            sol.copy(sol, dest, name=float_to_str(float(x_value)))

    def append_row(self, y_value, meta=None):
        const_params = deepcopy(self.const_params)
        const_params[self.y_param] = y_value
        filename = 'DATA/' + self.filename(self.phase, const_params, (self.x_param, ), meta, self.hf)
        source_file = h5py.File(filename)
        destination_file = h5py.File('DATA2D/' + self.this_filename())
        if float_to_str(y_value) in destination_file.keys():
            del(destination_file[float_to_str(y_value)])
        dest = destination_file.require_group(float_to_str(y_value))
        print('Appending data from ' + filename)
        for x, sol in source_file.items():
            sol.copy(sol, dest, name=float_to_str(float(x)))

    def export_col(self, x_value, meta=None):
        const_params = deepcopy(self.const_params)
        const_params[self.x_param] = x_value
        filename = 'DATA/' + self.filename(self.phase, const_params, (self.y_param, ), meta, self.hf)
        destination_file = h5py.File(filename, 'w')
        source_file = h5py.File('DATA2D/' + self.this_filename())
        print('Appending data to ' + filename)
        for y, row in source_file.items():
            if float_to_str(x_value) in row.keys():
                sol = row[float_to_str(x_value)]
                sol.copy(sol, destination_file, name=float_to_str(float(y)))

    def export_row(self, y_value, meta=None):
        const_params = deepcopy(self.const_params)
        const_params[self.y_param] = y_value
        filename = 'DATA/' + self.filename(self.phase, const_params, (self.x_param, ), meta, self.hf)
        destination_file = h5py.File(filename, 'w')
        source_file = h5py.File('DATA2D/' + self.this_filename())
        source = source_file[float_to_str(y_value)]
        print('Appending data to ' + filename)
        for x, sol in source.items():
            sol.copy(sol, destination_file, name=float_to_str(float(x)))




    def append_point(self, x_value, y_value, meta=None):
        const_params = deepcopy(self.const_params)
        const_params[self.x_param] = x_value
        const_params[self.y_param] = y_value
        filename = 'DATA/' + self.filename(self.phase, const_params, [], meta, self.hf)
        source_file = h5py.File(filename)
        destination_file = h5py.File('DATA2D/' + self.this_filename())
        group_name = float_to_str(y_value) + '/' + float_to_str(x_value)
        if group_name in destination_file.keys():
            del(destination_file[group_name])
        print('Appending data from ' + filename)
        source_file.copy(source_file, destination_file, name=group_name)



    def exploration2D(self, x_range, x_step, y_range, y_step,
                      call_on_solver, fast=False, vertical_neighbour=True, horizontal_neighbour=True):


        if x_range[0] <= x_range[1]:
            # ascending order
            x_grid = np.arange(x_range[0], x_range[1] + 0.5 * x_step, x_step)
            x_neighbour = lambda x: x - x_step
        else:
            # descending order
            x_grid = np.arange(x_range[1], x_range[0] + 0.5 * x_step, x_step)[::-1]
            x_neighbour = lambda x: x + x_step

        if y_range[0] <= y_range[1]:
            # ascending order
            y_grid = np.arange(y_range[0], y_range[1] + 0.5 * y_step, y_step)
            y_neighbour = lambda y: y - y_step
        else:
            # descending order
            y_grid = np.arange(y_range[1], y_range[0] + 0.5 * y_step, y_step)[::-1]
            y_neighbour = lambda y: y + y_step

        file = h5py.File('DATA2D/' + self.this_filename(), 'a')

        for y in y_grid[1:]:
            for x in x_grid[1:]:

                # Load solutions from neighbouring points
                neighbours = []
                if vertical_neighbour:
                    neighbours.append((x, y_neighbour(y)))
                if horizontal_neighbour:
                    neighbours.append((x_neighbour(x), y))

                neighbours_found = True
                for x_n, y_n in neighbours:
                    if float_to_str(y_n) not in file.keys():
                        neighbours_found = False
                        print (self.y_param + '=' + float_to_str(y_n) + ' not found')
                        break
                    elif float_to_str(x_n) not in file[float_to_str(y_n)].keys():
                        neighbours_found = False
                        print (self.x_param + '=' + float_to_str(x_n) + ' not found')
                        break

                if not neighbours_found:
                    print('Omitting the point (' + str(x) +', ' + str(y) + ')', flush=True)
                    continue

                def load_params_and_set_xy():
                    self.load_params(file, neighbours)
                    self.solver.set_model_param(x, self.x_param)
                    self.solver.set_model_param(y, self.y_param)
                    if self.phase in {'PM', 'FM'}:
                        rho = self.solver.get_rho()
                        for r in (2, 3, 6, 7):
                            rho[r] = 0
                        lam = self.solver.get_lambda()
                        for l in range(12, 18):
                            lam[l] = 0
                        eta = self.solver.get_eta()
                        for e in range(3, 5):
                            eta[e] = 0
                        self.solver.set_rho(rho)
                        self.solver.set_lambda(lam)
                        self.solver.set_eta(eta)

                load_params_and_set_xy()

                print('Starting optimization...', flush=True)
                print('(' + self.x_param + ', ' + self.y_param + ') = ' 
                      + '(' + str(x) + ', ' + str(y) + ')', flush=True)
                print(' ', flush=True)

                # Apply constraints, etc
                call_on_solver(self.solver)

                # Calculate the parameters
                if(not self.hf and fast):
                    success = self.solver.solve(mode = 'combined') == 0
                    if not success:
                        print('Trying once more using the iterative solver.', flush=True)
                        load_params_and_set_xy()
                        call_on_solver(self.solver)
                        success = self.solver.solve(mode = 'full') == 0
                        if not success:
                            print('We are stuck in the point (' + str(x) + ', ' + str(y) + '). Sorry.', flush=True)

                elif(not self.hf):
                    success = self.solver.solve(mode='full') == 0
                    if not success:
                        print('We are stuck in the point (' + str(x) +', ' + str(y) + '). Sorry.', flush=True)

                else:
                    success = self.solver.solve(mode='rho') == 0
                    if not success:
                        print('Trying once more using the iterative solver.', flush=True)
                        load_params_and_set_xy()
                        call_on_solver(self.solver)
                        self.solver.solve(mode='rho_iterations')
                        success = self.solver.solve(mode='rho') == 0
                        if not success:
                            print('We are stuck in the point (' + str(x) + ', ' + str(y) + '). Sorry.', flush=True)


                print(' ', flush=True)
                # Write solution to file
                if success:
                    self.this_write_to_file(file, x, y)
        file.close()



    def this_filename(self):
        var_params = (self.x_param, self.y_param)
        return self.filename(self.phase, self.const_params, var_params, self.meta, self.hf)

    @staticmethod
    def save_solution(solver, sol):
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
        sol['z'] = ()
        for spin in ('up', 'do'):
            sol['z'].attrs[spin] = solver.get_renormalization_factor(spin)

    @staticmethod
    def write_to_file(solver, f, x, y):
        if float_to_str(y) in f.keys():
            if float_to_str(x) in f[float_to_str(y)].keys():
                del(f[float_to_str(y)][float_to_str(x)])
        sol = f.require_group(float_to_str(y) + '/' + float_to_str(x))
        Explorer2D.save_solution(solver, sol)
        f.flush()

    def this_write_to_file(self, f, x, y):
        self.write_to_file(self.solver, f, x, y)


    @staticmethod
    def filename(phase, const_params, var_params, meta=None, hf=False, suffix=True):
        name = 'GA' if not hf else 'HF'
        name += '_' + phase
        for param, value in sorted(const_params.items(), key=lambda item: order.index(item[0])):
            name += '_' + param + float_to_str(value)
        for param in sorted(var_params, key=lambda p: order.index(p)):
            name += '_' + param
        if meta:
            name += '_' + meta
        return name + ('.hdf5' if suffix else '')

    @staticmethod
    def get_observable(param_name, sol):
        if param_name == 'energy':
            return sol['energy'].attrs['full']
        if param_name == 'z_up':
            return sol['z'].attrs['up']
        if param_name == 'z_do':
            return sol['z'].attrs['do']
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
            return Explorer2D.get_observable('m_f', sol) + Explorer2D.get_observable('m_c', sol)
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
    def load_params_from_file(solver, phase, const_params, var_param, var_param_value, meta=None, hf=False):
        filename = Explorer2D.filename(phase, const_params, (var_param, ), meta, hf)
        print(filename)
        f = h5py.File('DATA/' + filename)
        sol = f[float_to_str(var_param_value)]
        solver.set_rho(sol['rho'][:].tolist())
        solver.set_lambda(sol['lambda'][:].tolist())
        solver.set_eta(sol['eta'][:].tolist())
        for param in ('T', 'mu', 'e_f', 'V', 'U', 'U_p', 'J', 'J_c', 'n'):
            solver.set_model_param(sol['model'].attrs[param], param)
            
    def load_params(self, f, points):
        rho = np.zeros(8)
        mu = 0
        lam = np.zeros(18)
        eta = np.zeros(5)

        for x, y in points:
            sol = f[float_to_str(y) + '/' + float_to_str(x)]
            rho += sol['rho'][:] / len(points)
            mu += sol['model'].attrs['mu'] / len(points)
            lam += sol['lambda'][:] / len(points)
            eta += sol['eta'][:] / len(points)

        self.solver.set_rho(rho.tolist())
        self.solver.set_model_param(mu, 'mu')
        self.solver.set_lambda(lam.tolist())
        self.solver.set_eta(eta.tolist())


    def combine_files(self, metas, new_meta=None):
        self.meta = new_meta
        dest_file = h5py.File('DATA2D/' + self.this_filename())
        for meta in metas:
            self.meta = meta
            filename = 'DATA2D/' + self.this_filename()
            print(filename)
            file = h5py.File(filename, 'r')
            for y, row in file.items():
                for x, sol in row.items():
                    if y in dest_file.keys():
                        if x in dest_file[y].keys():
                            del(dest_file[y][x])
                    sol.copy(sol, dest_file, name=y + '/' + x)
            file.close()

    @staticmethod
    def load_params_from_solutions_file(solver, phase, const_params, meta, hf, energy):
        text = meta + '_solutions'
        f = h5py.File('SOL/' + Explorer2D.filename(phase, const_params, [], meta=text, hf=hf), 'r')
        sol = f['{0:0{width}}'.format(energy, width=4)]
        solver.set_rho(sol['rho'][:].tolist())
        solver.set_lambda(sol['lambda'][:].tolist())
        solver.set_eta(sol['eta'][:].tolist())
        for param in ('T', 'mu', 'e_f', 'V', 'U', 'U_p', 'J', 'J_c', 'n'):
            solver.set_model_param(sol['model'].attrs[param], param)





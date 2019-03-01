from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "GutzwillerSolver.h":
    cdef cppclass CPP_GutzwillerSolver "GutzwillerSolver":
        CPP_GutzwillerSolver() except +
        void set_phase(string, bool)
        void set_solver_params(double, double, double, double, double, double)
        void set_model_param(double, string)
        void set_dos(vector[double], vector[double])
        void set_rho(vector[double])
        void set_lambda(vector[double])
        void set_eta(vector[double])
        vector[double] get_rho()
        vector[double] get_lambda()
        vector[double] get_eta()
        double get_model_param(string)
        int solve(string)
        double get_uncorrelated_observable(string)
        double get_correlated_observable(string)
        double get_energy(string)
        vector[double] lambda_hessian_minors()
        void rho_calculate()
        void calculate_dos(double, double, double, double)
        vector[double] get_dos(string, string)
        vector[double] get_dos_energies()
        double get_renormalization_factor(string)
        vector[double] get_constraints()

cdef class GutzwillerSolver:
    cdef CPP_GutzwillerSolver *thisptr
    def __cinit__(self):
        self.thisptr = new CPP_GutzwillerSolver()
        self.n_error = 0.0001
    def __dealloc__(self):
        del self.thisptr

    def set_phase(self, str phase, bool hartree_fock = False):
        return self.thisptr.set_phase(phase.encode('utf-8'), hartree_fock)
    def set_solver_params(self, float ga_error, float ga_x, float rho_error, float rho_x, float lambda_error, float com_error, float n_error):
        self.n_error = n_error
        return self.thisptr.set_solver_params(ga_error, ga_x, rho_error, rho_x, lambda_error, com_error)
    def set_model_param(self, float value, str name):
        return self.thisptr.set_model_param(value, name.encode('utf-8'))
    def set_dos(self, list energies, list weights):
        return self.thisptr.set_dos(energies, weights)
    def set_rho(self, list rho):
        return self.thisptr.set_rho(rho)
    def set_lambda(self, list lambda_vec):
        return self.thisptr.set_lambda(lambda_vec)
    def set_eta(self, list eta):
        return self.thisptr.set_eta(eta)
    def get_rho(self):
        return self.thisptr.get_rho()
    def get_lambda(self):
        return self.thisptr.get_lambda()
    def get_eta(self):
        return self.thisptr.get_eta()
    def get_model_param(self, str name):
        return self.thisptr.get_model_param(name.encode('utf-8'))
    def solve(self, str mode = 'full'):
        return self.thisptr.solve(mode.encode('utf-8'))
    def get_uncorrelated_observable(self, str name):
        return self.thisptr.get_uncorrelated_observable(name.encode('utf-8'))
    def get_correlated_observable(self, str name):
        return self.thisptr.get_correlated_observable(name.encode('utf-8'))
    def get_energy(self, str part = 'full'):
        return self.thisptr.get_energy(part.encode('utf-8'))
    def lambda_hessian_minors(self):
        return self.thisptr.lambda_hessian_minors()
    def rho_calculate(self):
        return self.thisptr.rho_calculate()
    def calculate_dos(self, float E_min, float E_max, float step, float sigma):
        return self.thisptr.calculate_dos(E_min, E_max, step, sigma)
    def get_dos(self, str spin, str band = 'total'):
        return self.thisptr.get_dos_energies(), self.thisptr.get_dos(spin.encode('utf-8'), band.encode('utf-8'))
    def get_renormalization_factor(self, str spin):
        return self.thisptr.get_renormalization_factor(spin.encode('utf-8'))
    def get_constraints(self):
        return self.thisptr.get_constraints()

    cdef double n_error
    cdef double get_n(self):
        return 2 * (self.thisptr.get_correlated_observable('n_f_up'.encode('utf-8'))
                  + self.thisptr.get_correlated_observable('n_f_do'.encode('utf-8'))
                  + self.thisptr.get_uncorrelated_observable('n_c_up'.encode('utf-8'))
                  + self.thisptr.get_uncorrelated_observable('n_c_do'.encode('utf-8')))

    cdef void cpp_solve_fixing_n(self, string mode, double n, double[2] mu_bounds):
        # This procedure fails very often - sorry
        self.thisptr.set_model_param((mu_bounds[0] + mu_bounds[1]) / 2., 'mu'.encode('utf-8'))
        self.thisptr.solve(mode)
        print('mu: ' + str((mu_bounds[0] + mu_bounds[1]) / 2.), flush=True)
        print('n: ' + str(self.get_n()), flush=True)
        if self.get_n() - n > self.n_error:
            mu_bounds[1] = (mu_bounds[0] + mu_bounds[1]) / 2.
            self.cpp_solve_fixing_n(mode, n, mu_bounds)
        elif self.get_n() - n < -self.n_error:
            mu_bounds[0] = (mu_bounds[0] + mu_bounds[1]) / 2.
            self.cpp_solve_fixing_n(mode, n, mu_bounds)
        else:
            print('Success - mu has been found!', flush=True)
            return

    def solve_fixing_n(self, float n, list mu_bounds, str mode='rho'):
        cdef string mode_cpp = mode.encode('utf-8')
        cdef double[2] mu_bounds_cpp = mu_bounds
        self.thisptr.set_model_param(n, 'n'.encode('utf-8'))
        cdef double n_cpp = n
        self.cpp_solve_fixing_n(mode_cpp, n_cpp, mu_bounds_cpp)
        return (mu_bounds_cpp[0] + mu_bounds_cpp[1]) / 2.

    def get_total_filling(self):
        return self.get_n()



//
// Created by patryk on 19/04/16.
//

#ifndef GUTZWILLER_GUTZWILLERSOLVER_H
#define GUTZWILLER_GUTZWILLERSOLVER_H


#include "Model.h"
#include "LambdaSolver.h"
#include "RhoSolver.h"
#include "CombinedSolver.h"


class GutzwillerSolver {

public:
    GutzwillerSolver();
    void set_phase(string phase, bool hartree_fock = false);
    void set_solver_params(double ga_error, double ga_x, double rho_error, double rho_x, double lambda_error, double com_error);
    void set_model_param(double param_value, string param_name);
    void set_dos(vec energies, vec weights);
    void set_rho(vec rho);
    void set_lambda(vec lambda);
    void set_eta(vec eta);
    vec get_rho();
    vec get_lambda();
    vec get_eta();
    double get_model_param(string param_name);
    int solve(string mode = "full");
    double get_uncorrelated_observable(string param_name);
    double get_correlated_observable(string param_name);
    double get_energy(string part = "full");
    vec lambda_hessian_minors();
    void rho_calculate();
    void calculate_dos(double E_min, double E_max, double step, double sigma);
    vec get_dos(string spin, string band = "total");
    vec get_dos_energies();
    double get_renormalization_factor(string spin);
    vec get_constraints();

private:
    Model model;
    LambdaSolver lambda_solver;
    RhoSolver rho_solver;
    CombinedSolver combined_solver;

    const size_t iter_max = 1000;
    double error = 1e-4;
    double x = 0.5;
    inline bool are_equal_up_to_error(const vec &vec1, const vec &vec2);

    vec get_mixed_vec(const vec &old_vec, const vec &new_vec);

};


#endif //GUTZWILLER_GUTZWILLERSOLVER_H

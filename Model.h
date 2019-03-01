//
// Created by patryk on 25.12.15.
//

#ifndef GUTZWILLER_MODEL_H
#define GUTZWILLER_MODEL_H

#include <limits>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <armadillo>
#include <cassert>

using namespace std;

const size_t basis_size = 16;
const size_t n_lambda = 18;
const size_t n_eta = 5;
const size_t n_rho = 8;

typedef arma::Mat<double>::fixed<basis_size, basis_size> mat16;
typedef vector<double> vec;
typedef vector<size_t> uvec;


class Model {
public:
    Model();
    void update_rho(const vec &new_rho);
    void update_lambda(const vec &new_lambda);
    void update_eta(const vec &new_eta);

    void update_dos(const vec &energies, const vec &weights);
    void update_param(const string &param_name, double param_value);
    double get_param(const string &param_name);
    
    // Model parameters
    double T = 0.0001;
    double mu = 0;
    double e_f = 0;
    double V = 0;
    double U = 0;
    double U_p = 0;
    double J = 0;
    double J_c = 0;
    double n = 4; // filling [0, 8]

    // DOS
    vector<pair<double, double>> dos;

    // Rho, lambda, eta vectors
    vec rho;
    vec lambda;
    vec eta;

    // Local energy and energy gradient, hessian
    double local_f_energy();
    double local_hyb_energy();
    double local_energy();
    double local_energy_gradient(int l);
    double local_energy_hessian(int l_1, int l_2);

    // Constraints and constraints gradient, hessian
    vec constraints();
    vec constraints_gradient(int l);
    vec constraints_hessian(int l_1, int l_2);

    // Constraints functional and functional gradient, hessian
    double constraints_functional();
    double constraints_functional_gradient(int l);
    double constraints_functional_hessian(int l_1, int l_2);

    // Full Lagrange functional and functional gradient, hessian
    double lagrange_functional();
    double lagrange_functional_gradient(int l);
    double lagrange_functional_hessian(int l_1, int l_2);

    // Effective Hamiltonian parameters
    double e_f_up_eff();
    double e_f_do_eff();
    double V_up_eff();
    double V_do_eff();
    double Delta_f_up();
    double Delta_f_do();
    double Delta_f_up_cc();
    double Delta_f_do_cc();
    double Delta_fc_up();
    double Delta_fc_do();

    // Other observables
    double local_density_matrix_element(const string &param_name);

    vec all_constraints();



private:

    // Constant operator matrices
    mat16 N_1up;
    mat16 N_1do;
    mat16 A_op_up;
    mat16 A_op_do;
    mat16 F_cr_1up;
    mat16 F_cr_1do;
    mat16 F_cr_2up;
    mat16 F_cr_2do;
    void fill_N_1up(double *N_1up);
    void fill_N_1do(double *N_1do);
    void fill_A_op_up(double *A_op_up);
    void fill_A_op_do(double *A_op_do);
    void fill_F_cr_1up(double *F_cr_1up);
    void fill_F_cr_1do(double *F_cr_1do);
    void fill_F_cr_2up(double *F_cr_2up);
    void fill_F_cr_2do(double *F_cr_2do);

    // H_int model-dependent matrix
    mat16 H_int;
    void fill_H_int(double *H_int);

    // M and MC rho-dependent matrices
    mat16 M;
    mat16 MC_1up;
    mat16 MC_1do;
    mat16 MC_T_1up;
    mat16 MC_T_1do;
    void fill_M(double *M);
    void fill_MC_1up(double *MC_1up);
    void fill_MC_1do(double *MC_1do);
    void fill_MC_T_1up(double *MC_T_1up);
    void fill_MC_T_1do(double *MC_T_1do);

    // Gutzwiller lambda-dependent matrix
    mat16 P;
    vector<mat16> dP;
    void fill_P(double *P);


};


#endif //GUTZWILLER_MODEL_H

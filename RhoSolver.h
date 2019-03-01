//
// Created by patryk on 17.04.16.
//

#ifndef GUTZWILLER_RHOSOLVER_H
#define GUTZWILLER_RHOSOLVER_H

#include "Model.h"
#include "MultirootSolver.h"
//#include "omp.h"


typedef vector<double> vec;

class RhoSolver {

public:
    RhoSolver(Model &model);
    ~RhoSolver();
    int solve_by_iteration(const vec &rho_init);
    void set_error(double new_error);
    void set_mixing_parameter(double new_x);
    double c_energy();
    double get_local_density_matrix_element(const string &param_name);

    void set_mode(const string &mode, bool hf = false);
    int solve(const vec &rho_init, double mu_init);

    size_t n_eq;
    vec generate_args(const vec &rho, double mu);

    void update_model_rho_and_mu(const vec &args);
    void calculate_density_matrix(const vec &rho);


    vec return_F();

    bool hartree_fock;

    void calculate_dos(double E_min, double E_max, double step, double sigma);

    arma::vec dos_energies;
    arma::vec dos_up;
    arma::vec dos_do;

    arma::vec dos_f_up;
    arma::vec dos_f_do;
    arma::vec dos_c_up;
    arma::vec dos_c_do;


private:
    Model &model;

    arma::mat44 H_up;
    arma::mat44 H_do;
    arma::mat44 U_up;
    arma::mat44 U_do;
    arma::vec4 E_up;
    arma::vec4 E_do;
    arma::mat44 Rho_up;
    arma::mat44 Rho_do;

    inline double fermi(double x);

    void generate_H_up(arma::mat44 &H);
    void generate_H_do(arma::mat44 &H);
    void update_H(double en, arma::mat44 &H);
    void diagonalize_H(arma::vec4 &E, arma::mat44 &U, arma::mat44 &H);

    vec get_rho_from_density_matrix();
    vec get_mixed_rho();

    vec F(const vec &args);


    void print_state(size_t iter, const vec &rho_in, const vec &rho_out);

    double error = 1e-5;
    double x = 0.25;
    const size_t iter_max = 1000;




    string mode;

    MultirootSolver<RhoSolver> *solver = nullptr;
};


#endif //GUTZWILLER_RHOSOLVER_H

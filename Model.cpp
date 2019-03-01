//
// Created by patryk on 25.12.15.
//

#include "Model.h"

Model::Model() {

    // Initialize constant operator matrices
    N_1up.zeros();
    fill_N_1up(N_1up.memptr());
    N_1do.zeros();
    fill_N_1do(N_1do.memptr());
    A_op_up.zeros();
    fill_A_op_up(A_op_up.memptr());
    A_op_do.zeros();
    fill_A_op_do(A_op_do.memptr());
    F_cr_1up.zeros();
    fill_F_cr_1up(F_cr_1up.memptr());
    F_cr_1do.zeros();
    fill_F_cr_1do(F_cr_1do.memptr());
    F_cr_2up.zeros();
    fill_F_cr_2up(F_cr_2up.memptr());
    F_cr_2do.zeros();
    fill_F_cr_2do(F_cr_2do.memptr());
    // Remember to edit the cpp files - armadillo matrices are stored column by column!

    // Initialize P lambda derivatives
    lambda = vec(n_lambda, 0);
    dP = vector<mat16>(n_lambda, mat16());
    for(int l = 0; l < lambda.size(); l++){
        if(l > 0) lambda[l - 1] = 0;
        lambda[l] = 1;
        dP[l].zeros();
        fill_P(dP[l].memptr());
    }

    P.zeros();
    M.zeros();
    MC_1up.zeros();
    MC_1do.zeros();
    MC_T_1up.zeros();
    MC_T_1do.zeros();

    vec init_lambda = vec(n_lambda);
    for(size_t l = 0; l < 12; l++){
        init_lambda[l] = 1;
    }

    vec init_eta = vec(n_eta);
    vec init_rho = vec(n_rho);
    init_rho[0] = 0.5;
    init_rho[1] = 0.5;

    update_lambda(init_lambda);
    update_eta(init_eta);
    update_rho(init_rho);

    // Initialize H_int
    H_int.zeros();
    fill_H_int(H_int.memptr());
}

void Model::update_rho(const vec &new_rho) {
    rho = new_rho;
    fill_M(M.memptr());
    fill_MC_1up(MC_1up.memptr());
    fill_MC_1do(MC_1do.memptr());
    fill_MC_T_1up(MC_T_1up.memptr());
    fill_MC_T_1do(MC_T_1do.memptr());
}

void Model::update_lambda(const vec &new_lambda){
    lambda = new_lambda;
    fill_P(P.memptr());
}

void Model::update_eta(const vec &new_eta) {
    eta = new_eta;
}

void Model::update_dos(const vec &energies, const vec &weights) {
    assert(energies.size() == weights.size());
    dos.clear();
    for(size_t i = 0; i < energies.size(); i++){
        dos.push_back(make_pair(energies[i], weights[i]));
    }
}

void Model::update_param(const string &param_name, double param_value){
    if(param_name =="T")            T = param_value;
    else if(param_name == "mu")     mu = param_value;
    else if(param_name == "e_f")    e_f = param_value;
    else if(param_name == "V")      V = param_value;
    else if(param_name == "U")      U = param_value;
    else if(param_name == "U_p")    U_p = param_value;
    else if(param_name == "J")      J = param_value;
    else if(param_name == "J_c")    J_c = param_value;
    else if(param_name == "n")      n = param_value;
    else cout << "Wrong parameter name!" << endl;
    if(!(param_name == "mu")) fill_H_int(H_int.memptr());
}

double Model::get_param(const string &param_name){
    if(param_name =="T")            return T; 
    else if(param_name == "mu")     return mu;
    else if(param_name == "e_f")    return e_f;
    else if(param_name == "V")      return V;
    else if(param_name == "U")      return U;
    else if(param_name == "U_p")    return U_p;
    else if(param_name == "J")      return J;
    else if(param_name == "J_c")    return J_c;
    else if(param_name == "n")      return n;
    else cout << "Wrong parameter name!" << endl;
}


double Model::local_f_energy(){
    return arma::trace(P * (2 * (e_f - mu) * (N_1up + N_1do) + H_int) * P * M)
           + mu * 2 * (local_density_matrix_element("n_f_up") + local_density_matrix_element("n_f_do"));
}

double Model::local_hyb_energy(){
    return arma::trace(P * 4 * V * (F_cr_1up * P * MC_1up + F_cr_1do * P * MC_1do));
}

double Model::local_energy(){
    return local_f_energy() + local_hyb_energy();
}

double Model::local_energy_gradient(int l){
    return arma::trace(dP[l] * (2 * (e_f - mu) * (N_1up + N_1do) + H_int) * P * M)
           + arma::trace(P * (2 * (e_f - mu) * (N_1up + N_1do) + H_int) * dP[l] * M)
           + arma::trace(dP[l] * 4 * V * (F_cr_1up * P * MC_1up + F_cr_1do * P * MC_1do))
           + arma::trace(P * 4 * V * (F_cr_1up * dP[l] * MC_1up + F_cr_1do * dP[l] * MC_1do));
}

double Model::local_energy_hessian(int l_1, int l_2){
    return arma::trace(dP[l_1] * (2 * (e_f - mu) * (N_1up + N_1do) + H_int) * dP[l_2] * M)
           + arma::trace(dP[l_2] * (2 * (e_f - mu) * (N_1up + N_1do) + H_int) * dP[l_1] * M)
           + arma::trace(dP[l_1] * 4 * V
                         * (F_cr_1up * dP[l_2] * MC_1up + F_cr_1do * dP[l_2] * MC_1do))
           + arma::trace(dP[l_2] * 4 * V
                         * (F_cr_1up * dP[l_1] * MC_1up + F_cr_1do * dP[l_1] * MC_1do));
}

vec Model::constraints(){
    vec result(eta.size());
    mat16 PP = P * P;
    result[0] = arma::trace(PP * M) - 1;
    result[1] = arma::trace(PP * N_1up * M) - rho[0];
    result[2] = arma::trace(PP * N_1do * M) - rho[1];
    result[3] = 0.5 * arma::trace(PP * (A_op_up + A_op_up.t()) * M) - rho[2];
    result[4] = 0.5 * arma::trace(PP * (A_op_do + A_op_do.t()) * M) - rho[3];
    return result;
}

vec Model::constraints_gradient(int l){
    vec result(eta.size());
    mat16 dPP = dP[l] * P + P * dP[l];
    result[0] = arma::trace(dPP * M);
    result[1] = arma::trace(dPP * N_1up * M);
    result[2] = arma::trace(dPP * N_1do * M);
    result[3] = 0.5 * arma::trace(dPP * (A_op_up + A_op_up.t()) * M);
    result[4] = 0.5 * arma::trace(dPP * (A_op_do + A_op_do.t()) * M);
    return result;
}

vec Model::constraints_hessian(int l_1, int l_2){
    vec result(eta.size());
    mat16 ddPP = dP[l_1] * dP[l_2] + dP[l_2] * dP[l_1];
    result[0] = arma::trace(ddPP * M);
    result[1] = arma::trace(ddPP * N_1up * M);
    result[2] = arma::trace(ddPP * N_1do * M);
    result[3] = 0.5 * arma::trace(ddPP * (A_op_up + A_op_up.t()) * A_op_up * M);
    result[4] = 0.5 * arma::trace(ddPP * (A_op_do + A_op_do.t()) * A_op_do * M);
    return result;
}

double Model::constraints_functional(){
    vec constraints_vec = constraints();
    double result = 0;
    for(int n = 0; n < eta.size(); n++){
        result += eta[n] * constraints_vec[n];
    }
    return result;
}

double Model::constraints_functional_gradient(int l){
    vec gradient_vec = constraints_gradient(l);
    double result = 0;
    for(int n = 0; n < eta.size(); n++){
        result += eta[n] * gradient_vec[n];
    }
    return result;
}

double Model::constraints_functional_hessian(int l_1, int l_2){
    vec hessian_vec = constraints_hessian(l_1, l_2);
    double result = 0;
    for(int n = 0; n < eta.size(); n++){
        result += eta[n] * hessian_vec[n];
    }
    return result;
}

double Model::lagrange_functional(){
    return local_energy() + constraints_functional();
}

double Model::lagrange_functional_gradient(int l) {
    return local_energy_gradient(l) + constraints_functional_gradient(l);
}

double Model::lagrange_functional_hessian(int l_1, int l_2) {
    return local_energy_hessian(l_1, l_2) + constraints_functional_hessian(l_1, l_2);
}

double Model::local_density_matrix_element(const string &param_name){
    if(param_name == "n_f_up")          return arma::trace(P * N_1up * P * M);
    else if(param_name == "A_f_up")     return arma::trace(P * A_op_up * P * M);
    else if(param_name == "A_cf_up")    return arma::trace(P * F_cr_2up.t() * P * MC_1up);
    else if(param_name == "v_up")       return arma::trace(P * F_cr_1up * P * MC_1up);
    else if(param_name == "n_f_do")     return arma::trace(P * N_1do * P * M);
    else if(param_name == "A_f_do")     return arma::trace(P * A_op_do * P * M);
    else if(param_name == "A_cf_do")    return arma::trace(P * F_cr_2do.t() * P * MC_1do);
    else if(param_name == "v_do")       return arma::trace(P * F_cr_1do * P * MC_1do);
    else cout << "Wrong parameter name!" << endl;

}

vec Model::all_constraints() {
    vec result;
    mat16 PP = P * P;
    result.push_back(arma::trace(PP * M) - 1);
    result.push_back(arma::trace(PP * N_1up * M) - rho[0]);
    result.push_back(arma::trace(PP * N_1do * M) - rho[1]);
    result.push_back(arma::trace(PP * A_op_up * M) - rho[2]);
    result.push_back(arma::trace(PP * A_op_do * M) - rho[3]);
    result.push_back(arma::trace(PP * A_op_up.t() * M) - rho[2]);
    result.push_back(arma::trace(PP * A_op_do.t() * M) - rho[3]);
    result.push_back(arma::trace(N_1up * PP * M) - rho[0]);
    result.push_back(arma::trace(N_1do * PP * M) - rho[1]);
    result.push_back(arma::trace(A_op_up * PP * M) - rho[2]);
    result.push_back(arma::trace(A_op_do * PP * M) - rho[3]);
    result.push_back(arma::trace(A_op_up.t() * PP * M) - rho[2]);
    result.push_back(arma::trace(A_op_do.t() * PP * M) - rho[3]);
    result.push_back(Delta_f_up() - Delta_f_up_cc());
    result.push_back(Delta_f_do() - Delta_f_do_cc());
    return result;
}

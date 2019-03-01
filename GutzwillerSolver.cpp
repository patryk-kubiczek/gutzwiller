//
// Created by patryk on 19/04/16.
//

#include "GutzwillerSolver.h"

GutzwillerSolver::GutzwillerSolver() : rho_solver(model), lambda_solver(model), combined_solver(model, rho_solver, lambda_solver) {

}

void GutzwillerSolver::set_phase(string phase, bool hartree_fock) {
    lambda_solver.set_mode(phase);
    rho_solver.set_mode(phase, hartree_fock);
    combined_solver.set_mode(phase);
}

void GutzwillerSolver::set_solver_params(double ga_error, double ga_x, double rho_error, double rho_x, double lambda_error, double com_error) {
    error = ga_error;
    x = ga_x;
    rho_solver.set_error(rho_error);
    rho_solver.set_mixing_parameter(rho_x);
    lambda_solver.set_error(lambda_error);
    combined_solver.set_error(com_error);
}

void GutzwillerSolver::set_model_param(double param_value, string param_name) {
    model.update_param(param_name, param_value);
}

void GutzwillerSolver::set_dos(vec energies, vec weights) {
    model.update_dos(energies, weights);
}

void GutzwillerSolver::set_rho(vec rho) {
    model.update_rho(rho);
}

void GutzwillerSolver::set_lambda(vec lambda) {
    model.update_lambda(lambda);
}

void GutzwillerSolver::set_eta(vec eta) {
    model.update_eta(eta);
}

vec GutzwillerSolver::get_rho() {
    return model.rho;
}

vec GutzwillerSolver::get_lambda() {
    return model.lambda;
}

vec GutzwillerSolver::get_eta() {
    return model.eta;
}

double GutzwillerSolver::get_model_param(string param_name) {
    return model.get_param(param_name);
}

int GutzwillerSolver::solve(string mode) {
    if(mode == "iterations"){
        vec old_rho(n_rho);
        double old_mu;
        vec old_lambda(n_lambda);
        vec old_eta(n_eta);
        int status = 0;
        bool converged = false;
        for(size_t iter = 0; iter < iter_max; iter++){
            cout << "GLOBAL ITERATION: " << iter << endl;
            old_rho = model.rho;
            old_mu = model.mu;
            old_lambda = model.lambda;
            old_eta = model.eta;

            cout << "RHO solver:" << endl;
            rho_solver.solve(model.rho, model.mu);
            converged = are_equal_up_to_error(old_rho, model.rho) && abs(old_mu - model.mu) < error;
            model.update_rho(get_mixed_vec(old_rho, model.rho));
            model.mu = x * old_mu + (1 - x) * model.mu;

            cout << "LAMBDA solver:" << endl;
            status = lambda_solver.solve(model.lambda, model.eta);
            converged = converged && are_equal_up_to_error(old_lambda, model.lambda) && are_equal_up_to_error(old_eta, model.eta);
            model.update_lambda(get_mixed_vec(old_lambda, model.lambda));
            model.update_eta(get_mixed_vec(old_eta, model.eta));

            if(abs(model.mu) > 100){
                status = 100;
                break;
            }
            for(auto eta_el : model.eta){
                if(abs(eta_el) > 100){
                    status = 100;
                    break;
                }
            }

            if(converged){
                cout << "SUCCESS!" << endl;
                return 0;
            }
            cout << endl;
        }
        cout << "SORRY. NO CONVERGENCE!" << endl;
        return status;
    }
    else if(mode == "rho"){
        return rho_solver.solve(model.rho, model.mu);
    }
    else if(mode == "rho_iterations"){
        return rho_solver.solve_by_iteration(model.rho);
    }
    else if(mode == "lambda"){
        //lambda_solver.fix_n = false;
        return lambda_solver.solve(model.lambda, model.eta);
    }
    else if(mode == "combined"){
        return combined_solver.solve();
    }
    else if(mode == "full"){
        int status = solve("iterations");
        if(status != 0) return status;
        else return solve("combined");
    }
    else cout << "Wrong mode name!" << endl;
}

double GutzwillerSolver::get_uncorrelated_observable(string param_name){
    return rho_solver.get_local_density_matrix_element(param_name);
}

double GutzwillerSolver::get_correlated_observable(string param_name) {
    return model.local_density_matrix_element(param_name);
}


double GutzwillerSolver::get_energy(string part){
    if(part == "full"){
        return model.local_energy() + rho_solver.c_energy();
    }
    if(part == "f"){
        return model.local_f_energy();
    }
    if(part == "c"){
        return rho_solver.c_energy();
    }
    if(part == "fc"){
        return model.local_hyb_energy();
    }
    else cout << "Wrong name!";
}

bool GutzwillerSolver::are_equal_up_to_error(const vec &vec1, const vec &vec2) {
    assert(vec1.size() == vec2.size());
    for(size_t i = 0; i < vec1.size(); i++){
        if(abs(vec1[i] - vec2[i]) > error){
            return false;
        }
    }
    return true;
}

vec GutzwillerSolver::lambda_hessian_minors() {
    return lambda_solver.hessian_minors();
}


vec GutzwillerSolver::get_mixed_vec(const vec &old_vec, const vec &new_vec) {
    assert(old_vec.size() == new_vec.size());
    vec result;
    result.reserve(old_vec.size());
    for(size_t i = 0; i < old_vec.size(); i++){
        result.push_back((1 - x) * new_vec[i] + x * old_vec[i]);
    }
    return result;
}

void GutzwillerSolver::rho_calculate() {
    rho_solver.calculate_density_matrix(model.rho);

}

void GutzwillerSolver::calculate_dos(double E_min, double E_max, double step, double sigma) {
    rho_solver.calculate_dos(E_min, E_max, step, sigma);
}

vec GutzwillerSolver::get_dos_energies() {
    return arma::conv_to<vec>::from(rho_solver.dos_energies);

}

vec GutzwillerSolver::get_dos(string spin, string band) {
    if(band == "total"){
        if(spin == "up")
            return arma::conv_to<vec>::from(rho_solver.dos_up);
        if(spin == "do")
            return arma::conv_to<vec>::from(rho_solver.dos_do);
    }
    if(band == "f"){
        if(spin == "up")
            return arma::conv_to<vec>::from(rho_solver.dos_f_up);
        if(spin == "do")
            return arma::conv_to<vec>::from(rho_solver.dos_f_do);
    }
    if(band == "c"){
        if(spin == "up")
            return arma::conv_to<vec>::from(rho_solver.dos_c_up);
        if(spin == "do")
            return arma::conv_to<vec>::from(rho_solver.dos_c_do);
    }
}

double GutzwillerSolver::get_renormalization_factor(string spin) {
    double z = 0;
    if(spin == "up")
        z = model.V_up_eff() / model.V;
    if(spin == "do")
        z = model.V_do_eff() / model.V;
    return z * z;
}

vec GutzwillerSolver::get_constraints() {
    return model.all_constraints();
}



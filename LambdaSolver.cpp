//
// Created by patryk on 25.12.15.
//

#include "LambdaSolver.h"
LambdaSolver::LambdaSolver(Model &model) : model(model) {
    set_mode("FMSC");
}

LambdaSolver::~LambdaSolver() {
    delete solver;
    delete solver_no_jacobian;
}

void LambdaSolver::set_error(double error) {
    solver->error = error;
    solver_no_jacobian->error = error;
}

int LambdaSolver::solve(const vec &lambda_init, const vec &eta_init) {
    assert(lambda_init.size() == n_lambda);
    assert(eta_init.size() == n_eta);

    model.update_lambda(lambda_init);
    model.update_eta(eta_init);

    vec args(generate_args(model.lambda, model.eta));

    if(use_deriv) solver->solve(args, false);
    else solver_no_jacobian->solve(args, false);

    return (use_deriv ? solver->get_status() : solver_no_jacobian->get_status());
}

vec LambdaSolver::generate_args(const vec &lambda, const vec &eta){
    vec args;
    args.reserve(n_eq);

    for(auto l : lambda_indices){
        args.push_back(lambda[l]);
    }
    for(auto e : eta_indices){
        args.push_back(eta[e]);
    }
    return args;
}

void LambdaSolver::update_model_lambda_and_eta(const vec &args) {
    assert(args.size() == n_eq);

    for(size_t i_l = 0; i_l < lambda_indices.size(); i_l++){
        model.lambda[lambda_indices[i_l]] = args[i_l];
    }
    for(size_t i_e = 0; i_e < eta_indices.size(); i_e++){
        model.eta[eta_indices[i_e]] = args[lambda_indices.size() + i_e];
    }
    if(enforce_PM){
        model.lambda[2] = model.lambda[1];
        model.lambda[10] = model.lambda[9];

        model.lambda[13] = model.lambda[12];
        model.lambda[15] = model.lambda[14];
        model.lambda[17] = model.lambda[16];

        model.eta[2] = model.eta[1];
        model.eta[4] = model.eta[3];
    }
    if(bc_degeneracy){
        model.lambda[5] = model.lambda[4];
    }
    if(spin_triplet_degeneracy){
        model.lambda[7] = model.lambda[6];
        model.lambda[8] = model.lambda[6];
    }
    if(spin_triplet_partial_degeneracy){
        model.lambda[8] = model.lambda[6];
    }

    model.update_lambda(model.lambda);
    model.update_eta(model.eta);
}

vec LambdaSolver::return_F() {
    vec result;
    result.reserve(n_eq);
    for(auto l : lambda_indices){
        result.push_back(model.lagrange_functional_gradient(l));
    }
    vec constraints = model.constraints();
    for(auto e : eta_indices){
        result.push_back(constraints[e]);
    }
    return result;
}

vec LambdaSolver::return_dF() {
    vec result;
    result.reserve(n_eq * n_eq);
    vec constraints_grad;
    constraints_grad.reserve(n_eta);

    for(size_t i_l = 0; i_l < lambda_indices.size(); i_l++){
        for(size_t j_l = 0; j_l < lambda_indices.size(); j_l++){
            if(i_l <= j_l){
                result.push_back(model.lagrange_functional_hessian(lambda_indices[i_l], lambda_indices[j_l]));
            }
            else
                result.push_back(result[index(j_l, i_l)]);
        }
        constraints_grad =  model.constraints_gradient(lambda_indices[i_l]);
        for(auto e : eta_indices){
            result.push_back(constraints_grad[e]);
        }
    }
    for(size_t i_e = 0; i_e < eta_indices.size(); i_e++){
        for(size_t j_l = 0; j_l < lambda_indices.size(); j_l++){
            result.push_back(result[index(j_l, lambda_indices.size() + i_e)]);
        }
        for(size_t j_e = 0; j_e < eta_indices.size(); j_e++){
            result.push_back(0);
        }
    }
    return result;
}


vec LambdaSolver::F(const vec &args) {
    update_model_lambda_and_eta(args);
    return return_F();
}

vec LambdaSolver::dF(const vec &args) {
    update_model_lambda_and_eta(args);
    return return_dF();
}

pair<vec, vec> LambdaSolver::FdF(const vec &args) {
    update_model_lambda_and_eta(args);
    return make_pair(return_F(), return_dF());
}


size_t LambdaSolver::index(size_t i, size_t j) {
    return i * n_eq + j;
}


void LambdaSolver::set_mode(const string &new_mode) {
    mode = new_mode;
    if(mode == "FMSC") {
        initialize_indices(FMSC_indices);
        enforce_PM = false;
        bc_degeneracy = false;
        spin_triplet_degeneracy = false;
        spin_triplet_partial_degeneracy = false;
    }
    else if(mode == "FM") {
        initialize_indices(FM_indices);
        enforce_PM = false;
        bc_degeneracy = true;
        spin_triplet_degeneracy = false;
        spin_triplet_partial_degeneracy = false;
    }
    else if(mode == "PMSC") {
        initialize_indices(PMSC_indices);
        enforce_PM = true;
        bc_degeneracy = false;
        spin_triplet_degeneracy = false;
        spin_triplet_partial_degeneracy = true;
    }
    else if(mode == "PM") {
        initialize_indices(PM_indices);
        enforce_PM = true;
        bc_degeneracy = true;
        spin_triplet_degeneracy = true;
        spin_triplet_partial_degeneracy = false;
    }
    else cout << "Wrong mode name!" << endl;

    delete solver;
    delete solver_no_jacobian;
    solver = new MultirootSolverWithJacobian<LambdaSolver>(this, &LambdaSolver::F, &LambdaSolver::dF, &LambdaSolver::FdF,
                                                           n_eq);
    solver_no_jacobian = new MultirootSolver<LambdaSolver>(this, &LambdaSolver::F, n_eq);
}

void LambdaSolver::initialize_indices(const uvec &indices) {
    n_eq = indices.size();
    lambda_indices.clear();
    eta_indices.clear();
    for(auto i : indices)
        if(i < n_lambda){
            lambda_indices.push_back(i);
        }
        else{
            eta_indices.push_back(i - n_lambda);
        }
}

vec LambdaSolver::hessian_minors() {
    vec minors;
    arma::mat hessian(return_dF());
    hessian.reshape(n_eq, n_eq);
    for(size_t i = 2 * eta_indices.size() + 1; i < n_eq; i++){
        minors.push_back(arma::det(hessian.submat(n_eq - 1 - i, n_eq - 1 - i, n_eq - 1, n_eq - 1)));
    }
    return minors;
}



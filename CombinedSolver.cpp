//
// Created by patryk on 22/05/16.
//

#include "CombinedSolver.h"

CombinedSolver::CombinedSolver(Model &model, RhoSolver &rho_solver, LambdaSolver &lambda_solver)
        : model(model), rho_solver(rho_solver), lambda_solver(lambda_solver){
    set_mode("FMSC");
}

CombinedSolver::~CombinedSolver() {
    delete solver;
}

void CombinedSolver::set_error(double error) {
    solver->error = error;
}

int CombinedSolver::solve() {
    vec rho_args(rho_solver.generate_args(model.rho, model.mu));
    vec lambda_args(lambda_solver.generate_args(model.lambda, model.eta));
    vec args;
    args.reserve(rho_n_eq + lambda_n_eq);
    args.insert(args.end(), rho_args.begin(), rho_args.end());
    args.insert(args.end(), lambda_args.begin(), lambda_args.end());

    solver->solve(args, false);
    return solver->get_status();
}

void CombinedSolver::set_mode(const string &mode) {
    rho_solver.set_mode(mode, rho_solver.hartree_fock);
    lambda_solver.set_mode(mode);

    rho_n_eq = rho_solver.n_eq;
    lambda_n_eq = lambda_solver.n_eq;

    delete solver;
    solver = new MultirootSolver<CombinedSolver>(this, &CombinedSolver::F, rho_n_eq + lambda_n_eq);

}

vec CombinedSolver::F(const vec &args) {
    assert(args.size() == rho_n_eq + lambda_n_eq);

    vec rho_vec;
    rho_vec.reserve(rho_n_eq);
    for(size_t r = 0; r < rho_n_eq; r++){
        rho_vec.push_back(args[r]);
    }
    vec lambda_vec;
    lambda_vec.reserve(lambda_n_eq);
    for(size_t l = 0; l < lambda_n_eq; l++){
        lambda_vec.push_back(args[rho_n_eq + l]);
    }

    rho_solver.update_model_rho_and_mu(rho_vec);
    lambda_solver.update_model_lambda_and_eta(lambda_vec);
    rho_solver.calculate_density_matrix(model.rho);

    lambda_vec = lambda_solver.return_F();
    rho_vec = rho_solver.return_F();

    vec result;
    result.reserve(rho_n_eq + lambda_n_eq);

    result.insert(result.end(), rho_vec.begin(), rho_vec.end());
    result.insert(result.end(), lambda_vec.begin(), lambda_vec.end());

    return result;
}

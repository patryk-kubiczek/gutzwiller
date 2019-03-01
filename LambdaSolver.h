//
// Created by patryk on 25.12.15.
//

#ifndef GUTZWILLER_LAMBDASOLVER_H
#define GUTZWILLER_LAMBDASOLVER_H

#include "Model.h"
#include "MultirootSolver.h"
#include "RhoSolver.h"


class LambdaSolver {
public:
    LambdaSolver(Model &model);
    ~LambdaSolver();

    void set_error(double error);
    int solve(const vec &lambda_init, const vec &eta_init);
    void set_mode(const string &mode);
    vec hessian_minors();

    const bool use_deriv = true;

    size_t n_eq;
    vec generate_args(const vec &lambda, const vec &eta);

    void update_model_lambda_and_eta(const vec &args);
    vec return_F();


private:
    Model &model;

    void initialize_indices(const uvec &indices);
    inline size_t index(size_t i, size_t j);

    const uvec FMSC_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,    18, 19, 20, 21, 22};
    const uvec FM_indices =   {0, 1, 2, 3, 4,    6, 7, 8, 9, 10, 11,                            18, 19, 20        };
    const uvec PMSC_indices = {0, 1,    3, 4, 5, 6, 7,    9,     11, 12,     14,     16,        18, 19,     21    };
    const uvec PM_indices =   {0, 1,    3, 4,    6,       9,     11,                            18, 19,           };

    // Note: d_s is assumed = d_E in normal state

    vec return_dF();
    vec F(const vec &args);
    vec dF(const vec &args);
    pair<vec, vec> FdF(const vec &args);

    string mode;
    vec lambda_indices;
    vec eta_indices;
    bool enforce_PM;
    bool bc_degeneracy;
    bool spin_triplet_degeneracy;
    bool spin_triplet_partial_degeneracy;

    MultirootSolverWithJacobian<LambdaSolver> *solver = nullptr;
    MultirootSolver<LambdaSolver> *solver_no_jacobian = nullptr;

};


#endif //GUTZWILLER_LAMBDASOLVER_H

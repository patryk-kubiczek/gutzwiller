//
// Created by patryk on 22/05/16.
//

#ifndef GUTZWILLER_COMBINEDSOLVER_H
#define GUTZWILLER_COMBINEDSOLVER_H

#include "Model.h"
#include "MultirootSolver.h"
#include "RhoSolver.h"
#include "LambdaSolver.h"

class CombinedSolver {

public:
    CombinedSolver(Model &model, RhoSolver &rho_solver, LambdaSolver &lambda_solver);
    ~CombinedSolver();

    void set_error(double error);
    int solve();
    void set_mode(const string &mode);


private:
    Model &model;
    RhoSolver &rho_solver;
    LambdaSolver &lambda_solver;

    vec F(const vec &args);

    size_t lambda_n_eq;
    size_t rho_n_eq;

    MultirootSolver<CombinedSolver> *solver = nullptr;
};


#endif //GUTZWILLER_COMBINEDSOLVER_H

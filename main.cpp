#include <iostream>
#include <iomanip>
#include <random>


#include "Model.h"
#include "LambdaSolver.h"
#include "RhoSolver.h"


using namespace std;

int main() {

    Model model;
    RhoSolver rho_solver(model);
    LambdaSolver lambda_solver(model, <#initializer#>);

    random_device r;
    mt19937 generator(r());
    normal_distribution<> normal_dist(0, 0.01);


    vector<double> init_lambda =  {0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0};
    vector<double> init_eta = {1, 1, 1, 1, 1, 1, 1};

    model.update_lambda(init_lambda);
    model.update_eta(init_eta);

//    auto PP = model.P * model.P;
//    auto M = model.M;
//    cout << PP << endl;
//    cout << M << endl;
//    cout << arma::trace(PP * M) - 1 << endl;
//    cout << arma::trace(PP * model.N_1up * M) - model.rho[0] << endl;
//    cout << arma::trace(PP * model.N_1do * M) - model.rho[1] << endl;
//    cout << arma::trace(PP * model.A_op_up * M) - model.rho[2] << endl;
//    cout << arma::trace(PP * model.A_op_do * M) - model.rho[3] << endl;
//    cout << arma::trace(PP * model.A_op_up.t() * M) - model.rho[2] << endl;
//    cout << arma::trace(PP * model.A_op_do.t() * M) - model.rho[3] << endl;
    cout << model.local_energy() << endl;
    cout << model.constraints_functional() << endl;
    cout << model.lagrange_functional() << endl;

    for(auto &val : init_lambda){
        val += normal_dist(generator);
        cout << setw(8) << val << "\t";
    }
    for(auto &val : init_eta){
        val += normal_dist(generator);
        cout << setw(8) << val << "\t";
    }
    for(int i = 12; i < 12+8;  i++){
        init_lambda[i] = 0;
    }

//    cout << arma::trace(model.P * model.P * model.M ) << endl;
    cout << model.local_energy() << endl;
    cout << model.constraints_functional() << endl;
    cout << model.lagrange_functional() << endl;


    lambda_solver.solve(init_lambda, init_eta);
//    cout << model.P << endl;
    for(auto eta : model.eta){
        cout << setw(6) << eta << "\t";
    }
    cout << endl;
//    cout << arma::trace(model.P * model.P * model.M ) << endl;
    cout << model.local_energy() << endl;
    cout << model.constraints_functional() << endl;
    cout << model.lagrange_functional() << endl;

//    cout << PP << endl;
//    cout << M << endl;


}



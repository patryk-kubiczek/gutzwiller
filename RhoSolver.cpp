//
// Created by patryk on 17.04.16.
//

#include "RhoSolver.h"


RhoSolver::RhoSolver(Model &model) : model(model) {
    set_mode("FMSC");
}

RhoSolver::~RhoSolver() {
    delete solver;
}

int RhoSolver::solve_by_iteration(const vec &rho_init) {
    calculate_density_matrix(rho_init);
    vec new_rho = get_rho_from_density_matrix();
    for(size_t iter = 0; iter < iter_max; iter++){
        print_state(iter, model.rho, new_rho);
        for(size_t i = 0; i < n_rho; i++){
            if(abs(new_rho[i] - model.rho[i]) > error) goto next_iteration;
        }
        model.rho = new_rho;
        cout << "Success!" << endl;
        return 0;
        next_iteration:
            calculate_density_matrix(get_mixed_rho());
            new_rho = get_rho_from_density_matrix();
    }
    cout << "Sorry. No convergence." << endl;
    return 1;
}

void RhoSolver::set_error(double new_error) {
    error = new_error;
    solver->error = error;
}

void RhoSolver::set_mixing_parameter(double new_x){
    x = new_x;
}


void RhoSolver::generate_H_up(arma::mat44 &H) {

    double e_f_eff = model.e_f_up_eff();
    double Delta_f_eff = model.Delta_f_up();
    double V_eff = model.V_up_eff();
    double Delta_cf_eff = model.Delta_fc_up();

    H =    {{0,             0,              V_eff,          Delta_cf_eff   },
            {0,             0,              Delta_cf_eff,  -V_eff          },
            {V_eff,         Delta_cf_eff,   e_f_eff,        Delta_f_eff    },
            {Delta_cf_eff,  -V_eff,         Delta_f_eff,   -e_f_eff        }};
}

void RhoSolver::generate_H_do(arma::mat44 &H) {

    double e_f_eff = model.e_f_do_eff();
    double Delta_f_eff = model.Delta_f_do();
    double V_eff = model.V_do_eff();
    double Delta_cf_eff = model.Delta_fc_do();

    H =    {{0,             0,              V_eff,          Delta_cf_eff   },
            {0,             0,              Delta_cf_eff,  -V_eff          },
            {V_eff,         Delta_cf_eff,   e_f_eff,        Delta_f_eff    },
            {Delta_cf_eff,  -V_eff,         Delta_f_eff,   -e_f_eff        }};
}

void RhoSolver::update_H(double en, arma::mat44 &H) {
    double e_c = en - model.mu;
    H(0, 0) = e_c;
    H(1, 1) = -e_c;
}

void RhoSolver::diagonalize_H(arma::vec4 &E, arma::mat44 &U, arma::mat44 &H) {
    arma::eig_sym(E, U, H);
}

void RhoSolver::calculate_density_matrix(const vec &rho) {
    model.update_rho(rho);
    Rho_up.zeros();
    Rho_do.zeros();
    generate_H_up(H_up);
    generate_H_do(H_do);
    for(auto point : model.dos) {
        update_H(point.first, H_up);
        update_H(point.first, H_do);
        diagonalize_H(E_up, U_up, H_up);
        diagonalize_H(E_do, U_do, H_do);
        E_up.transform([this](double x) { return fermi(x); });
        E_do.transform([this](double x) { return fermi(x); });
        Rho_up += point.second * U_up * arma::diagmat(E_up) * U_up.t();
        Rho_do += point.second * U_do * arma::diagmat(E_do) * U_do.t();
    }

    if(mode == "PM" || mode == "FM") {
        // Make all anomalous averages exactly zero
        Rho_up(0, 3) = 0;
        Rho_up(1, 2) = 0;
        Rho_up(2, 1) = 0;
        Rho_up(3, 0) = 0;
        Rho_up(2, 3) = 0;
        Rho_up(3, 2) = 0;

        Rho_do(0, 3) = 0;
        Rho_do(1, 2) = 0;
        Rho_do(2, 1) = 0;
        Rho_do(3, 0) = 0;
        Rho_do(2, 3) = 0;
        Rho_do(3, 2) = 0;
    }
    if(mode == "PM" || mode == "PMSC") {
        // Enforce paramagnetism
        Rho_up = (Rho_up + Rho_do) / 2.;
        Rho_do = (Rho_up + Rho_do) / 2.;
    }
}

double RhoSolver::c_energy() {
    double c_en_up = 0;
    double c_en_do = 0;
    generate_H_up(H_up);
    generate_H_do(H_do);
    for(auto point : model.dos) {
        update_H(point.first, H_up);
        update_H(point.first, H_do);
        diagonalize_H(E_up, U_up, H_up);
        diagonalize_H(E_do, U_do, H_do);
        E_up.transform([this](double x) { return fermi(x); });
        E_do.transform([this](double x) { return fermi(x); });
        c_en_up += point.second * (point.first - model.mu) * arma::mat44(U_up * arma::diagmat(E_up) * U_up.t())(0, 0);
        c_en_do += point.second * (point.first - model.mu) * arma::mat44(U_do * arma::diagmat(E_do) * U_do.t())(0, 0);
    }
    return 2 * (c_en_up + c_en_do)
           + model.mu * 2 * (get_local_density_matrix_element("n_c_up") + get_local_density_matrix_element("n_c_do"));
}

double RhoSolver::get_local_density_matrix_element(const string &param_name){
    if(param_name == "n_c_up")          return Rho_up(0, 0);
    else if(param_name == "n_f_up")     return Rho_up(2, 2);
    else if(param_name == "A_c_up")     return Rho_up(0, 1);
    else if(param_name == "A_cf_up")    return Rho_up(0, 3);
    else if(param_name == "A_f_up")     return Rho_up(2, 3);
    else if(param_name == "v_up")       return Rho_up(0, 2);
    else if(param_name == "n_c_do")     return Rho_do(0, 0);
    else if(param_name == "n_f_do")     return Rho_do(2, 2);
    else if(param_name == "A_c_do")     return Rho_do(0, 1);
    else if(param_name == "A_cf_do")    return Rho_do(0, 3);
    else if(param_name == "A_f_do")     return Rho_do(2, 3);
    else if(param_name == "v_do")       return Rho_do(0, 2);
    else cout << "Wrong parameter name!" << endl;
}





double RhoSolver::fermi(double x) {
    double exponential = exp(x / model.T);
    double result = (exponential != HUGE_VAL ? 1. / (exponential + 1) : 0);
    return result;
}

vec RhoSolver::get_rho_from_density_matrix() {
    vec result;
    result.reserve(n_rho);
    result.push_back(Rho_up(2, 2));
    result.push_back(Rho_do(2, 2));
    result.push_back(Rho_up(2, 3));
    result.push_back(Rho_do(2, 3));
    result.push_back(Rho_up(0, 2));
    result.push_back(Rho_do(0, 2));
    result.push_back(Rho_up(0, 3));
    result.push_back(Rho_do(0, 3));
    return result;
}

vec RhoSolver::get_mixed_rho() {
    vec result = get_rho_from_density_matrix();
    if(x != 0){
        for(size_t i = 0; i < n_rho; i++){
            result[i] = (1 - x) * result[i] + x * model.rho[i];
        }
    }
    return result;
}

void RhoSolver::print_state(size_t iter, const vec &rho_in, const vec &rho_out) {
    cout << "Iteration: " << iter << ", X_in: [";
    for(size_t i = 0; i < rho_in.size(); i++){
        cout << setprecision(4) << fixed << rho_in[i];
        if(i < rho_in.size() - 1) cout << ", ";
    }
    cout << "], X_out: [";
    for(size_t i = 0; i < rho_out.size(); i++){
        cout << setprecision(4) << fixed << rho_out[i];
        if(i < rho_out.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
}


void RhoSolver::set_mode(const string &new_mode, bool hf) {
    mode = new_mode;
    hartree_fock = hf;
    if(mode == "FMSC") {
        n_eq = 9;
        if(hartree_fock) n_eq = 5;
    }
    else if(mode == "FM") {
        n_eq = 5;
        if(hartree_fock) n_eq = 3;
    }
    else if(mode == "PMSC") {
        n_eq = 5;
        if(hartree_fock) n_eq = 3;
    }
    else if(mode == "PM") {
        n_eq = 3;
        if(hartree_fock) n_eq = 2;
    }
    else cout << "Wrong mode name!" << endl;

    delete solver;
    solver = new MultirootSolver<RhoSolver>(this, &RhoSolver::F, n_eq);

}

void RhoSolver::update_model_rho_and_mu(const vec &args) {
    assert(args.size() == n_eq);

    if(mode == "FMSC") {
        // args: 0: n_f_up, 1: n_f_do, 2: A_up, 3: A_do, 4: v_up, 5: v_do, 6: C_up, 7: C_do, 8: mu
        for(size_t r = 0; r < (!hartree_fock ? model.rho.size() : 4); r++){
            model.rho[r] = args[r];
        }
    }
    else if(mode == "FM") {
        // args: 0: n_f_up, 1: n_f_do, 2: v_up, 3: v_do, 4: mu
        model.rho[0] = args[0];
        model.rho[1] = args[1];

        if(!hartree_fock){
            model.rho[4] = args[2];
            model.rho[5] = args[3];
        }
    }
    else if(mode == "PMSC") {
        // args: 0: n_f, 1: A, 2: v, 3: C, 4: mu
        model.rho[0] = args[0];
        model.rho[1] = args[0];
        model.rho[2] = args[1];
        model.rho[3] = args[1];

        if(!hartree_fock){
            model.rho[4] = args[2];
            model.rho[5] = args[2];
            model.rho[6] = args[3];
            model.rho[7] = args[3];
        }
    }
    else if(mode == "PM") {
        // args: 0: n_f, 1: v, 2: mu
        model.rho[0] = args[0];
        model.rho[1] = args[0];

        if(!hartree_fock){
            model.rho[4] = args[1];
            model.rho[5] = args[1];
        }
    }
    model.mu = args.back();

    //model.update_rho(model.rho);
}

vec RhoSolver::return_F() {
    vec result;
    result.reserve(n_eq);
    vec calculated_rho = get_rho_from_density_matrix();
    if(mode == "FMSC") {
        // args: 0: n_f_up, 1: n_f_do, 2: A_up, 3: A_do, 4: v_up, 5: v_do, 6: C_up, 7: C_do, 8: mu
        for(size_t r = 0; r < (!hartree_fock ? model.rho.size() : 4); r++){
            result.push_back(calculated_rho[r] - model.rho[r]);
        }
    }
    else if(mode == "FM") {
        // args: 0: n_f_up, 1: n_f_do, 2: v_up, 3: v_do, 4: mu
        result.push_back(calculated_rho[0] - model.rho[0]);
        result.push_back(calculated_rho[1] - model.rho[1]);

        if(!hartree_fock) {
            result.push_back(calculated_rho[4] - model.rho[4]);
            result.push_back(calculated_rho[5] - model.rho[5]);
        }
    }
    else if(mode == "PMSC") {
        // args: 0: n_f, 1: A, 2: v, 3: C, 4: mu
        result.push_back(calculated_rho[0] - model.rho[0]);
        result.push_back(calculated_rho[2] - model.rho[2]);

        if(!hartree_fock) {
            result.push_back(calculated_rho[4] - model.rho[4]);
            result.push_back(calculated_rho[6] - model.rho[6]);
        }
    }
    else if(mode == "PM") {
        // args: 0: n_f, 1: v, 2: mu
        result.push_back(calculated_rho[0] - model.rho[0]);

        if(!hartree_fock) {
            result.push_back(calculated_rho[4] - model.rho[4]);
        }
    }
    if(hartree_fock){
        for(size_t r = 4; r < model.rho.size(); r++){
            model.rho[r] = calculated_rho[r];
        }
    }
    double calculated_n = 2 * (model.local_density_matrix_element("n_f_up") + model.local_density_matrix_element("n_f_do") 
                          + get_local_density_matrix_element("n_c_up") + get_local_density_matrix_element("n_c_do"));
    result.push_back(calculated_n - model.n);
    return result;
}

vec RhoSolver::F(const vec &args) {
    assert(args.size() == n_eq);
    update_model_rho_and_mu(args);
    calculate_density_matrix(model.rho);
    return return_F();
}

int RhoSolver::solve(const vec &rho_init, double mu_init) {
    assert(rho_init.size() == n_rho);

    model.update_rho(rho_init);
    model.update_param("mu", mu_init);

    vec args(generate_args(model.rho, model.mu));

    solver->solve(args, false);
    if(hartree_fock) model.rho = get_rho_from_density_matrix();

    return solver->get_status();
}

vec RhoSolver::generate_args(const vec &rho, double mu) {
    vec args;
    args.reserve(n_eq);

    if(mode == "FMSC") {
        // args: 0: n_f_up, 1: n_f_do, 2: A_up, 3: A_do, 4: v_up, 5: v_do, 6: C_up, 7: C_do, 8: mu
        for(size_t r = 0; r < (!hartree_fock ? model.rho.size() : 4); r++){
            args.push_back(rho[r]);
        }
    }
    else if(mode == "FM") {
        // args: 0: n_f_up, 1: n_f_do, 2: v_up, 3: v_do, 4: mu
        args.push_back(rho[0]);
        args.push_back(rho[1]);
        if(!hartree_fock) {
            args.push_back(rho[4]);
            args.push_back(rho[5]);
        }
    }
    else if(mode == "PMSC") {
        // args: 0: n_f, 1: A, 2: v, 3: C, 4: mu
        args.push_back(rho[0]);
        args.push_back(rho[2]);
        if(!hartree_fock) {
            args.push_back(rho[4]);
            args.push_back(rho[6]);
        }
    }
    else if(mode == "PM") {
        // args: 0: n_f, 1: v, 2: mu
        args.push_back(rho[0]);
        if(!hartree_fock) {
            args.push_back(rho[4]);
        }
    }
    args.push_back(mu);
    return args;
}

void RhoSolver::calculate_dos(double E_min, double E_max, double step, double sigma) {

    auto gaussian = [](double x, double sigma) {return exp(-x * x / (2 * sigma * sigma)) / (sqrt(2 * M_PI) * sigma);};

    size_t N = (size_t) ((E_max - E_min) / step);

    dos_energies.zeros(N);
    dos_up.zeros(N);
    dos_do.zeros(N);
    dos_f_up.zeros(N);
    dos_f_do.zeros(N);
    dos_c_up.zeros(N);
    dos_c_do.zeros(N);

    for(size_t i = 0; i < N; i++){
        dos_energies(i) = E_min + i * step;
    }


    generate_H_up(H_up);
    generate_H_do(H_do);

    arma::uword max_pos;
    double contribution;
    double norm;

    for(auto point : model.dos) {
        update_H(point.first, H_up);
        update_H(point.first, H_do);
        diagonalize_H(E_up, U_up, H_up);
        diagonalize_H(E_do, U_do, H_do);
        for(size_t I = 0; I < 4; I++){
            arma::abs(U_up.col(I)).max(max_pos);
            if(max_pos == 0 || max_pos == 2) {
                for (size_t i = 0; i < N; i++) {
                    contribution = 2 * point.second * gaussian(dos_energies(i) - E_up(I), sigma);
                    norm = arma::norm(U_up.col(I));
                    dos_up(i) += contribution;
                    dos_f_up(i) += (U_up(2, I) * U_up(2, I) + U_up(3, I) * U_up(3, I)) / norm * contribution;
                    dos_c_up(i) +=(U_up(0, I) * U_up(0, I) + U_up(1, I) * U_up(1, I)) / norm * contribution;
                }
            }
            arma::abs(U_do.col(I)).max(max_pos);
            if(max_pos == 0 || max_pos == 2) {
                for (size_t i = 0; i < N; i++) {
                    dos_do(i) += 2 * point.second * gaussian(dos_energies(i) - E_do(I), sigma);
                }
            }
        }
    }



}

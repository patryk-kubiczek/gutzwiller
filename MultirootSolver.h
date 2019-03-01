#ifndef MULTIROOTSOLVER_H
#define MULTIROOTSOLVER_H

#include <vector>
#include <utility>
#include <iostream>
#include <iomanip>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_matrix.h>

using namespace std;
const size_t number_precision = 6;

template <class C>
struct f_params
{
    f_params(C *object_ptr, vector<double> (C::*F)(const vector<double> &)) : object_ptr(object_ptr), F(F) {};
    C* object_ptr;
    vector<double> (C::*F)(const vector<double>&);
};

template<class C>
class MultirootSolver{
public:
    MultirootSolver(C* object_ptr, vector<double> (C::*F)(const vector<double>&), size_t n);
    ~MultirootSolver();

    void solve(vector<double> &v_init, bool print = 0);
    vector<double> solution();
    int get_status();

    double error = 1e-7;
    const size_t iter_max = 1000;

private:
    static int function(const gsl_vector *x, void *params, gsl_vector *f);
    void print_state (size_t iter, gsl_multiroot_fsolver * s);

    f_params<C> p;
    const size_t n;

    const gsl_multiroot_fsolver_type *T;
    gsl_multiroot_fsolver *s;
    gsl_multiroot_function f;
    gsl_vector *x;

    int status;
};


// Implementation

template <class C>
MultirootSolver<C>::MultirootSolver(C* object_ptr, vector<double> (C::*F)(const vector<double>&), size_t n)
        : p(object_ptr, F), n(n){

    T = gsl_multiroot_fsolver_hybrids;
    s = gsl_multiroot_fsolver_alloc(T, n);
    f = {&function, n, &p};
    x = gsl_vector_alloc(n);
}

template <class C>
MultirootSolver<C>::~MultirootSolver() {
    gsl_multiroot_fsolver_free(s);
    gsl_vector_free(x);
}

template <class C>
void MultirootSolver<C>::solve(vector<double> &v_init, bool print) {

    gsl_vector_view v = gsl_vector_view_array(v_init.data(), v_init.size());
    gsl_vector_memcpy (x, &v.vector);

    gsl_multiroot_fsolver_set(s, &f, x);

    size_t iter = 0;
    if(print) print_state(iter, s);
    do
    {
        iter++;
        status = gsl_multiroot_fsolver_iterate(s);
        if(print) print_state(iter, s);
        if (status)   /* check if solver is stuck */
            break;
        status = gsl_multiroot_test_residual(s->f, error);
        //status = gsl_multiroot_test_delta(s->dx, s->x, error, error);
    }
    while (status == GSL_CONTINUE && iter < iter_max);
    if(print == 0) print_state(iter, s);
    cout << "Status: " << gsl_strerror(status) << endl;
}

template <class C>
vector<double> MultirootSolver<C>::solution() {
    return vector<double>(s->x->data, s->x->data + s->x->size);
}

template <class C>
int MultirootSolver<C>::get_status() {
    return status;
}

template <class C>
int MultirootSolver<C>::function(const gsl_vector *x, void *params, gsl_vector *f) {
    f_params<C>* p = static_cast<f_params<C>*>(params);

    vector<double> arguments(x->data, x->data + x->size);
    vector<double> values = ((p->object_ptr)->*(p->F))(arguments);

    gsl_vector_view v = gsl_vector_view_array(values.data(), values.size());
    gsl_vector_memcpy(f, &v.vector);

    return GSL_SUCCESS;
}

template <class C>
void MultirootSolver<C>::print_state(size_t iter, gsl_multiroot_fsolver *s) {
    cout << "Iteration: " << iter << ", X: [";
    for(size_t i = 0; i < s->x->size; i++){
        cout << setprecision(number_precision) << fixed << gsl_vector_get(s->x, i);
        if(i < s->x->size - 1) cout << ", ";
    }
    cout << "], F(X): [";
    for(size_t i = 0; i < s->f->size; i++){
        cout << setprecision(number_precision) << fixed << gsl_vector_get(s->f, i);
        if(i < s->x->size - 1) cout << ", ";
    }
    cout << "]" << endl;
}

//With Jacobian

template <class C>
struct fdf_params
{
    fdf_params(C* object_ptr, vector<double> (C::*F)(const vector<double>&), vector<double> (C::*dF)(const vector<double>&),
               pair<vector<double>, vector<double> > (C::*FdF)(const vector<double>&), size_t n)
            : object_ptr(object_ptr), F(F), dF(dF), FdF(FdF), n(n) {};
    C* object_ptr;
    vector<double> (C::*F)(const vector<double>&);
    vector<double> (C::*dF)(const vector<double>&);
    pair<vector<double>, vector<double> > (C::*FdF)(const vector<double>&);
    const size_t n;
};

template<class C>
class MultirootSolverWithJacobian{
public:
    MultirootSolverWithJacobian(C* object_ptr, vector<double> (C::*F)(const vector<double>&),
                                vector<double> (C::*dF)(const vector<double>&),
                                pair<vector<double>, vector<double> > (C::*FdF)(const vector<double>&),
                                size_t n);
    ~MultirootSolverWithJacobian();

    void solve(vector<double> &v_init, bool print = 0);
    vector<double> solution();
    int get_status();

    double error = 1e-7;
    const size_t iter_max = 1000;

private:
    static int function(const gsl_vector *x, void *params, gsl_vector *f);
    static int jacobian(const gsl_vector *x, void *params, gsl_matrix *J);
    static int func_and_jac(const gsl_vector *x, void *params, gsl_vector *f, gsl_matrix *J);

    void print_state (size_t iter, gsl_multiroot_fdfsolver * s);

    fdf_params<C> p;
    const size_t n;

    int status;

    const gsl_multiroot_fdfsolver_type *T;
    gsl_multiroot_fdfsolver *s;
    gsl_multiroot_function_fdf fdf;
    gsl_vector *x;
};

// Implementation

template <class C>
MultirootSolverWithJacobian<C>::MultirootSolverWithJacobian(C* object_ptr, vector<double> (C::*F)(const vector<double>&),
                                                            vector<double> (C::*dF)(const vector<double>&),
                                                            pair<vector<double>, vector<double> > (C::*FdF)(const vector<double>&),
                                                            size_t n)
        : p(object_ptr, F, dF, FdF, n), n(n){

    T = gsl_multiroot_fdfsolver_hybridsj;
    s = gsl_multiroot_fdfsolver_alloc(T, n);
    fdf = {&function, &jacobian, &func_and_jac, n, &p};
    x = gsl_vector_alloc(n);
}

template <class C>
MultirootSolverWithJacobian<C>::~MultirootSolverWithJacobian(){
    gsl_multiroot_fdfsolver_free(s);
    gsl_vector_free(x);
}

template <class C>
void MultirootSolverWithJacobian<C>::solve(vector<double> &v_init, bool print) {

    gsl_vector_view v = gsl_vector_view_array(v_init.data(), v_init.size());
    gsl_vector_memcpy(x, &v.vector);

    gsl_multiroot_fdfsolver_set(s, &fdf, x);

    int status;
    size_t iter = 0;
    if(print) print_state(iter, s);
    do
    {
        iter++;
        status = gsl_multiroot_fdfsolver_iterate(s);
        if(print) print_state(iter, s);
        if (status)   /* check if solver is stuck */
            break;
        status = gsl_multiroot_test_residual(s->f, error);
        //status = gsl_multiroot_test_delta(s->dx, s->x, error, error);
    }
    while (status == GSL_CONTINUE && iter < iter_max);
    if(print == 0) print_state(iter, s);
    cout << "Status: " << gsl_strerror(status) << endl;
}

template <class C>
vector<double> MultirootSolverWithJacobian<C>::solution() {
    return vector<double>(s->x->data, s->x->data + s->x->size);
}

template <class C>
int MultirootSolverWithJacobian<C>::get_status() {
    return status;
}

template <class C>
int MultirootSolverWithJacobian<C>::function(const gsl_vector *x, void *params, gsl_vector *f) {
    fdf_params<C>* p = static_cast<fdf_params<C>*>(params);

    vector<double> arguments(x->data, x->data + x->size);
    vector<double> values = ((p->object_ptr)->*(p->F))(arguments);

    gsl_vector_view v = gsl_vector_view_array(values.data(), p->n);
    gsl_vector_memcpy(f, &v.vector);

    return GSL_SUCCESS;
}

template <class C>
int MultirootSolverWithJacobian<C>::jacobian(const gsl_vector *x, void *params, gsl_matrix *J) {
    fdf_params<C>* p = static_cast<fdf_params<C>*>(params);

    vector<double> arguments(x->data, x->data + x->size);
    vector<double> jacobian = ((p->object_ptr)->*(p->dF))(arguments);

    gsl_matrix_view v = gsl_matrix_view_array(jacobian.data(), p->n, p->n);
    gsl_matrix_memcpy(J, &v.matrix);

    return GSL_SUCCESS;
}

template <class C>
int MultirootSolverWithJacobian<C>::func_and_jac(const gsl_vector *x, void *params, gsl_vector *f, gsl_matrix *J) {
    fdf_params<C>* p = static_cast<fdf_params<C>*>(params);

    vector<double> arguments(x->data, x->data + x->size);
    pair<vector<double>, vector<double> > pair = ((p->object_ptr)->*(p->FdF))(arguments);

    gsl_vector_view v1 = gsl_vector_view_array(pair.first.data(), p->n);
    gsl_vector_memcpy(f, &v1.vector);

    gsl_matrix_view v2 = gsl_matrix_view_array(pair.second.data(), p->n, p->n);
    gsl_matrix_memcpy(J, &v2.matrix);

    return GSL_SUCCESS;
}

template <class C>
void MultirootSolverWithJacobian<C>::print_state(size_t iter, gsl_multiroot_fdfsolver *s) {
    cout << "Iteration: " << iter << ", X: [";
    for(size_t i = 0; i < s->x->size; i++){
        cout << setprecision(number_precision) << fixed << gsl_vector_get(s->x, i);
        if(i < s->x->size - 1) cout << ", ";
    }
    cout << "], F(X): [";
    for(size_t i = 0; i < s->f->size; i++){
        cout << setprecision(number_precision) << fixed << gsl_vector_get(s->f, i);
        if(i < s->x->size - 1) cout << ", ";
    }
    cout << "]" << endl;
}


#endif //MULTIROOTSOLVER_H

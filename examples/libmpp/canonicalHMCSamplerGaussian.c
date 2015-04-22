/**
 * \file canonicalHMCSamplerGaussian.c
 *
 * This file explains how to compile an example code using mpp headers
 *
 * You will need mpp.h header file and the library libmpp.a compiled
 * and ready to link
 *
 * You can compile this example by
 *
 * gcc -pedantic -Wall -Wextra -Wfatal-errors -std=c11 -g0 -O3 -Ipath_to_libmpp_headers canonicalHMCSamplerGaussian.c -o example_libmpp_canonicalHMCSamplerGaussian -Lpath_to_mpp_lib -Lpath_to_boost_lib -Lpath_to_pthread_lib -lboost_serialization -lboost_filesystem -lboost_system -lboost_log -lboost_thread -lboost_date_time -lpthread -lstdc++
 *
 * For example in my desktop, I will compile this by
 *
 * gcc  -pedantic -Wall -Wextra -Wfatal-errors -std=c11 -g0 -O3 -I/arxiv/libraries/ubuntu/gcc/mpp/include canonicalHMCSamplerGaussian.c -o example_lbmpp_canonicalHMCSamplerGaussian -L/arxiv/libraries/ubuntu/gcc/mpp/lib -L/usr/lib/x86_64-linux-gnu -lmpp -lboost_serialization -lboost_filesystem -lboost_system -lboost_log -lboost_thread -lboost_date_time -lpthread -lstdc++
 *
 * This will create an executable example_lbmpp_canonicalHMCSamplerGaussian
 *
 * You also need to set LD_LIBRARY_PATH appropriately. For example,
 *
 * export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/arxiv/libraries/ubuntu/gcc/mpp/lib
 *
 */

// std headers
#include <stdlib.h>

// include the mpp header file
#include <mpp.h>


// define the dimensionality of the Gaussian posterior
long const num_params = 1000;

// define the mean of the distribution
double * mu;
// define the invese of the diagonal covariance matrix, i.e, 1/var(i)
double * sigma_inv;

// define log-posterior function, value is returned to val
void log_post_func(double const* q, double* val)
{
    double log_post = 0;
    for(long i=0;i<num_params;++i)
    {
        log_post += (mu[i] - q[i])*(mu[i] - q[i])*sigma_inv[i];
    }
    *val = -0.5*log_post;
}

// define the derivatives of the log-post wrt q
void log_post_derivs(double const* q, double* dq)
{
    for(long i=0;i<num_params;++i)
    {
        dq[i] = sigma_inv[i]*(mu[i] - q[i]);
    }
}


int main(void)
{
    // allocate and assign values to mu and sigma inverse
    sigma_inv = (double*) malloc(num_params * sizeof(double));
    mu = (double*) malloc(num_params * sizeof(double));
    for(long i=0;i<num_params;++i)
    {
        sigma_inv[i] = 1.;
        mu[i] = 0.;
    }

    // maximum value of the parameter epsilon; 0<epsilon<2
    double const max_eps = 1.;
    // maximum number of leapfrog / euler steps
    long const max_num_steps = 10;
    // starting point for the sampling
    double * start_point = (double*) malloc(num_params * sizeof(double));
    for(long i=0;i<num_params;++i)
    {
        start_point[i] = 0.;
    }
    // seed for the random number generator
    long const rand_seed = 1234;
    // number samples to be taken in one interation.
    // how often you would like the chains to be written?
    // every time packet_size samples
    long const packet_size = 100;
    // number samples to be burned
    long const num_burn = 0;
    // number samples to be taken (after burning)
    long const num_samples = 1000;
    // path to chains and other output
    char const* root_path_str = "./example_libmpp_hmc_diag_canon";
    // do we require output to console? 0 means NO, !=0 means YES
    int const console_output = 1;
    // delimiter for the chain data
    char const* delimiter = ",";
    // percision with which the chains should be written
    int const precision = 10;
    // inverse of the diagonal kinetic energy mass matrix
    // this should be close/equal to the invese of the covariance
    // matrix of parameters / posterior distriubtion if Gaussian
    // If parameters are correlated, try diagonal elemens
    double *  ke_diag_m_inv = (double*) malloc(num_params * sizeof(double));
    for(long i=0;i<num_params;++i)
    {
        ke_diag_m_inv[i] = 1.;
    }

    // call mpp interface for double; prefix is d_
    d_mpp_hmc_diag_canon(
        &num_params,
        &max_eps,
        &max_num_steps,
        start_point,
        &rand_seed,
        &packet_size,
        &num_burn,
        &num_samples,
        root_path_str,
        &console_output,
        delimiter,
        &precision,
        ke_diag_m_inv,
        &log_post_func,
        &log_post_derivs
    );

    // free memroy
    free(sigma_inv);
    free(mu);
    free(start_point);
    free(ke_diag_m_inv);

    return 0;
}

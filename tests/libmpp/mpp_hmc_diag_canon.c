#include <stdlib.h>

#include <mpp.h>


long const num_params = 10;

double * mu;
double * sigma_inv;

void log_post_func(double const* q, double* val)
{
    double log_post = 0;
    for(long i=0;i<num_params;++i)
    {
        log_post += (mu[i] - q[i])*(mu[i] - q[i])*sigma_inv[i];
    }
    *val = -0.5*log_post;
}

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
    char const* root_path_str = "./test_mpp_hmc_diag_canon";
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

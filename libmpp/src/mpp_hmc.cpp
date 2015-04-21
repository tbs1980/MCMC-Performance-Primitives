#include <mpp/core>
#include <mpp.h>


class mppLogPost
{
public:
    typedef double realScalarType;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> realVectorType;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> realDiagMatrixType;
    typedef realVectorType::Index indexType;

    mppLogPost(
        void (*p_log_post_func)( double const*,double*),
        void (*p_log_post_derivs)(double const*,double*),
        indexType const numDims
        )
    {
        m_p_log_post_func = p_log_post_func;
        m_p_log_post_derivs = p_log_post_derivs;
        m_numDims = numDims;
    }

    inline void value(realVectorType  const & q, double & val) const
    {
        double const*  p_q = &q(0);
        m_p_log_post_func(p_q,&val);
    }

    inline void derivs(realVectorType  const & q,realVectorType & dq) const
    {
        double const*  p_q = &q(0);
        double * p_dq = &dq(0);
        m_p_log_post_derivs(p_q,p_dq);
    }

    inline indexType numDims(void) const
    {
        return m_numDims;
    }

private:
    void (*m_p_log_post_func)( double const* ,double*);
    void (*m_p_log_post_derivs)(double const* ,double*);
    indexType m_numDims;
};

void mpp_hmc_diag_canon(
    long const* num_params,
    double const* max_eps,
    long const* max_num_steps,
    double const* start_point,
    long const* rand_seed,
    long const* packet_size,
    long const* num_burn,
    long const* num_samples,
    char const* root_path_str,
    int const* console_output,
    char const* delimiter,
    int const* precision,
    double const*  ke_diag_m_inv,
    void (*p_log_post_func)(double const* ,double*),
    void (*p_log_post_derivs)(double const* ,double*)
    )
{
    typedef double realScalarType;
    typedef mppLogPost potEngType;
    typedef mpp::Hamiltonian::GaussKineticEnergyDiag<realScalarType> kinEngType;
    typedef mpp::utils::randomSTD<realScalarType> rVGenType;
    typedef mpp::Hamiltonian::leapfrog leapfrogIntegratorPolicy;
    typedef mpp::Hamiltonian::canonicalHMC<rVGenType,potEngType,kinEngType,
        leapfrogIntegratorPolicy> canonicalHMCType;
    typedef typename potEngType::realVectorType realVectorType;
    typedef typename kinEngType::realDiagMatrixType realDiagMatrixType;
    typedef typename realDiagMatrixType::Index indexType;
    typedef typename rVGenType::seedType seedType;
    // define control
    typedef mpp::control::finiteSamplesControl<realScalarType> controlType;
    // define IO
    typedef mpp::IO::IOWriteAllParams<realScalarType> IOType;
    // define the sampler
    typedef mpp::sampler::canonicalMCMCSampler<canonicalHMCType,controlType,IOType>
        samplerType;

    // define the diemensionaligy of the problem
    size_t const numParams = (size_t) (*num_params);

    // make  posterior distribution
    potEngType G(p_log_post_func,p_log_post_derivs,(indexType) numParams);

    // define a kinetic energy type
    realDiagMatrixType mInv(numParams);
    for(indexType i=0;i<(indexType)numParams;++i)
    {
        mInv(i) = ke_diag_m_inv[i];
    }
    kinEngType K(mInv);

    // define the step size and the number of steps for the integrator
    realScalarType const maxEps = (realScalarType) (*max_eps);
    indexType const maxNumsteps = (indexType) (*max_num_steps);

    // define the start point
    realVectorType startPoint(numParams);
    for(indexType i=0;i<(indexType)numParams;++i)
    {
        startPoint(i) = start_point[i];
    }

    // define a random number seed
    seedType seed = (seedType) (*rand_seed);

    // define the Hamiltonian Monte Carlo
    canonicalHMCType canonHMC(maxEps,maxNumsteps,startPoint,seed,G,K);

    // define the finite samples control
    size_t const packetSize = (size_t) (*packet_size);
    size_t const numBurn = (size_t) (*num_burn);
    size_t const numSamples = (size_t) (*num_samples);
    std::string const rootPathStr(root_path_str);
    bool const consoleOutput = (*console_output) > 0 ? true : false;

    std::string randState = canonHMC.getRandState();

    controlType ctrl(numParams, packetSize, numBurn, numSamples, rootPathStr,
        consoleOutput,startPoint,randState);

    // define IO
    const std::string outFileName = ctrl.getChainFileName();
    const std::string delim(delimiter);
    const unsigned int prec = (unsigned int) (*precision);

    IOType writeAllIO(outFileName,delim,prec);

    // define the sampler
    samplerType::run(canonHMC,ctrl,writeAllIO);
}

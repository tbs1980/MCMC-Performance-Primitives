void MPP_FLOATING_POINT_TYPE_FUNC_NAME(mpp_hmc_diag_canon)(
    long const* num_params,
    MPP_FLOATING_POINT_TYPE const* max_eps,
    long const* max_num_steps,
    MPP_FLOATING_POINT_TYPE const* start_point,
    long const* rand_seed,
    long const* packet_size,
    long const* num_burn,
    long const* num_samples,
    char const* root_path_str,
    int const* console_output,
    char const* delimiter,
    int const* precision,
    MPP_FLOATING_POINT_TYPE const*  ke_diag_m_inv,
    void (*p_log_post_func)(MPP_FLOATING_POINT_TYPE const* ,MPP_FLOATING_POINT_TYPE*),
    void (*p_log_post_derivs)(MPP_FLOATING_POINT_TYPE const* ,MPP_FLOATING_POINT_TYPE*)
    )
{
    // define the log posterior type
    typedef MPP_FLOATING_POINT_TYPE realScalarType;
    typedef mppLogPost<realScalarType> logPosteriorType;
    typedef logPosteriorType::indexType indexType;

    // define the canonical Hamiltonian sampler interface
    typedef mpp::canonicalHamiltonianSampler<logPosteriorType> samplerType;

    typedef typename samplerType::realVectorType realVectorType;

    typedef typename logPosteriorType::realDiagMatrixType realDiagMatrixType;

    typedef typename samplerType::seedType seedType;

    // define the diemensionaligy of the problem
    size_t const numParams = (size_t) (*num_params);

    // make  posterior distribution
    logPosteriorType G(p_log_post_func,p_log_post_derivs,(indexType) numParams);

    // define a kinetic energy type
    realDiagMatrixType MInv(numParams);
    for(indexType i=0;i<(indexType)numParams;++i)
    {
        MInv(i) = ke_diag_m_inv[i];
    }

    // define the step size and the number of steps for the integrator
    realScalarType const maxEps = (realScalarType) (*max_eps);
    indexType const maxNumSteps = (indexType) (*max_num_steps);

    // define the start point
    realVectorType startPoint(numParams);
    for(indexType i=0;i<(indexType)numParams;++i)
    {
        startPoint(i) = start_point[i];
    }

    // define a random number seed
    seedType randSeed = (seedType) (*rand_seed);

    // define the finite samples control
    size_t const packetSize = (size_t) (*packet_size);
    size_t const numBurn = (size_t) (*num_burn);
    size_t const numSamples = (size_t) (*num_samples);
    std::string const rootPathStr(root_path_str);
    bool const consoleOutput = (*console_output) != 0 ? true : false;

    const std::string delim(delimiter);
    const unsigned int prec = (unsigned int) (*precision);

    // define the sampler
    samplerType canonHamiltSampler(
        G,
        numParams,
        maxEps,
        maxNumSteps,
        startPoint,
        randSeed,
        packetSize,
        numBurn,
        numSamples,
        rootPathStr,
        consoleOutput,
        delim,
        prec,
        MInv
    );

    // finally run the sampler
    canonHamiltSampler.run();

}

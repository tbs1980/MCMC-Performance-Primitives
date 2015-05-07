#ifndef MPP_CANONICAL_HAMILTONIAN_SAMPLER_HPP
#define MPP_CANONICAL_HAMILTONIAN_SAMPLER_HPP

namespace mpp
{
    /**
     * \class canonicalHamiltonianSampler
     *
     * \brief User interface for the simplest Hamiltonian Sampler
     *
     * \tparam logPosteriorType log-Posterior type
     *
     * This class provides an easy to use interface for the canonical
     * Hamiltonian samler. The interface requires the user to provide a class
     * with log posterior and its gradietns. All parameters will be written
     * into the chain file. The sampler stops after required number samples are
     * taken after burning the specified number of samples.
     */
    template<class logPosteriorType>
    class canonicalHamiltonianSampler
    {
    public:
        // define the log posterior type

        /**
         * \typedef typename logPosteriorType::realScalarType realScalarType
         * \brief log-posterior type
         */
        typedef typename logPosteriorType::realScalarType realScalarType;

        /**
         * \typedef mpp::utils::randomSTD<realScalarType> rVGenType
         * \brief random variate generator type
         */
        typedef mpp::utils::randomSTD<realScalarType> rVGenType;

        // define the engine type

        /**
         * \typedef mpp::Hamiltonian::GaussKineticEnergyDiag<realScalarType> kinEngType
         * \brief the kinetic energy type
         */
        typedef mpp::Hamiltonian::GaussKineticEnergyDiag<realScalarType> kinEngType;

        /**
         * \typedef mpp::Hamiltonian::leapfrog leapfrogIntegratorPolicy;
         * \brief leapfrog integration of the Hamiltonian
         */
        typedef mpp::Hamiltonian::leapfrog leapfrogIntegratorPolicy;

        /**
         * \typedef mpp::Hamiltonian::canonicalHMC<rVGenType,potEngType,kinEngType,
             leapfrogIntegratorPolicy> canonicalHMCType
         * \brief canonical Hamiltonian Monte Carlo
         */
        typedef mpp::Hamiltonian::canonicalHMC<rVGenType,logPosteriorType,kinEngType,
            leapfrogIntegratorPolicy> canonicalHMCType;

        // define the matrix and vector types

        /**
         * \typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType
         * \brief real floating point vector
         */
        typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType;

        /**
         * \typedef typename kinEngType::realDiagMatrixType realDiagMatrixType
         * \breif the diagonal matrix type
         */
        typedef typename kinEngType::realDiagMatrixType realDiagMatrixType;

        // define the seed type for the random number generator

        /**
         * \typedef typename rVGenType::seedType seedType
         * \brief defines the seed type for the random number generator
         */
        typedef typename rVGenType::seedType seedType;

        // define the controller type

        /**
         * \typedef mpp::control::finiteSamplesControl<realScalarType> controlType
         * \brief define the controller type for the sampler
         */
        typedef mpp::control::finiteSamplesControl<realScalarType> controlType;

        // define the IO type

        /**
         * \typedef mpp::IO::IOWriteAllParams<realScalarType> IOType
         * \brief define the IO type of the sampler
         */
        typedef mpp::IO::IOWriteAllParams<realScalarType> IOType;

        // define the sampler type

        /**
         * \typedef mpp::sampler::canonicalMCMCSampler<canonicalHMCType,controlType,IOType> samplerType
         * \brief define the sampler type
         */
        typedef mpp::sampler::canonicalMCMCSampler<canonicalHMCType,controlType,IOType> samplerType;

        // for safety we define some upper limits
        static size_t const maxNumParams = 10000000;/**< maximum number of parameters (dimensions) */
        static size_t const maxMaxNumSteps = 100; /**< maximum of the allowed number steps in the leapforg */
        static size_t const maxPacketSize = 10000; /**< maximum packet size */
        static unsigned int const maxPrecision = 100; /**< maximum allowed precision */


        /**
         * \brief A constructor that sets up the sampler
         *
         * \param logPost log-posterior object
         * \param numParams number of parameters in the posterior
         * \param maxEps maximum value of the epsilon of leapfrog
         * \param maxNumSteps maximum number steps in the leapfrog
         * \param startPoint staring point of the sampler
         * \param randSeed random number seed
         * \param packetSize number samples to be geerated in one go of HMC
         * \param numBurn number of samples to be burned
         * \param numSamples number of samples to be taken after burning
         * \param rootPathStr path-prefix to output files
         * \param consoleOutput do we need output to console?
         * \param delimiter delimiter of the chain file
         * \param precision precision of the cahin file
         * \param keDiagMInv diagonal elements of the inverse mass matrix
         */
        canonicalHamiltonianSampler(
            logPosteriorType & logPost,
            size_t const numParams,
            realScalarType const maxEps,
            size_t const maxNumSteps,
            realVectorType const & startPoint,
            size_t const randSeed,
            size_t const packetSize,
            size_t const numBurn,
            size_t const numSamples,
            std::string const & rootPathStr,
            bool const consoleOutput,
            std::string const & delimiter,
            unsigned int const & precision,
            realVectorType const & keDiagMInv
        )
        :mLogPost(logPost),mNumParams(numParams),mMaxEps(maxEps)
        ,mMaxNumSteps(maxNumSteps),mStartPoint(startPoint)  ,mRandSeed(randSeed)
        ,mPacketSize(packetSize),mNumBurn(numBurn),mNumSamples(numSamples)
        ,mRootPathStr(rootPathStr),mConsoleOutput(consoleOutput)
        ,mDelimiter(delimiter),mPrecision(precision),mKeDiagMInv(keDiagMInv)
        {
            BOOST_ASSERT_MSG(
                numParams > 0 and numParams < maxNumParams ,
                "Number of parameters should be a positive number which is less than maxNumParams set at the begining of canonicalHamiltonianSampler class. If higher number of of parameters are required, this class needs to be recompiled with appropriate maxNumParams."
            );

            BOOST_ASSERT_MSG(
                numParams == (size_t) logPost.numDims(),
                "Number of Dimensions in the logPost class should be indentical to the number of parameters in the sampler."
            );

            BOOST_ASSERT_MSG(
                maxEps > 0 and maxEps < realScalarType(2),
                "For stability of Leapforg discretisation, the value of epsilon should be between 0 and 2."
            );

            BOOST_ASSERT_MSG(
                maxNumSteps > 0 and maxNumSteps < maxMaxNumSteps,
                "Maximum number of steps in the Leapfrog discretisation should be a positive number which is less than maxMaxNumSteps set at the beginning of canonicalHamiltonianSampler class. If higher number steps are required, this class needs to be recompiled with appropriate maxMaxNumSteps."
            );

            BOOST_ASSERT_MSG(
                numParams == (size_t) startPoint.rows(),
                "The diemsionality of the start point vector should be identical to the number of parameters in the Hamiltonian sampler."
            );

            BOOST_ASSERT_MSG(
                packetSize > 0 and packetSize < maxPacketSize,
                "Packet-size for the control class should be a positive number which is less than maxPacketSize set at the beginning of canonicalHamiltonianSampler class. If higher packet-size is required, this class needs to be recompiled with appropriate maxPacketSize."
            );

            BOOST_ASSERT_MSG(
                numSamples > 0,
                "Number of samples should be a postive number"
            );

            BOOST_ASSERT_MSG(
                precision > 0 and precision < maxPrecision,
                "Precision for the IO class should be a positive number which is less than maxPrecision set at the beginning of canonicalHamiltonianSampler class. If higher precision is required, this class needs to be recompiled with appropriate maxPrecision."
            );

            BOOST_ASSERT_MSG(
                numParams == (size_t) keDiagMInv.rows(),
                "The diemsionality of the diagonal inverse mass matrix should be identical to the number of parameters in the Hamiltonian sampler."
            );

            // test if the root path is good or not
            std::string tempFileName = rootPathStr + std::string("_mppTempFile.dtxt");
            std::ofstream ofs(tempFileName);

            BOOST_ASSERT_MSG(
                ofs.is_open(),
                "Error in root path provided. Please check if the path has write access."
            );

            if(ofs.is_open())
            {
                ofs.close();
                boost::filesystem::remove(tempFileName);
            }
        }

        /**
         * \brief Run the Hamiltonian Sampler
         */
        void run()
        {
            kinEngType K(mKeDiagMInv);

            // define the Hamiltonian Monte Carlo
            canonicalHMCType canonHMC(mMaxEps,mMaxNumSteps,mStartPoint,mRandSeed,mLogPost,K);

            // get the current random number state for control
            std::string randState = canonHMC.getRandState();

            // define the control object
            controlType ctrl(mNumParams, mPacketSize, mNumBurn, mNumSamples, mRootPathStr,
                mConsoleOutput,mStartPoint,randState);

            // define IO object
            const std::string outFileName = ctrl.getChainFileName();
            IOType writeAllIO(outFileName,mDelimiter,mPrecision);

            // run the sampler
            samplerType::run(canonHMC,ctrl,writeAllIO);
        }

    private:

        logPosteriorType & mLogPost;/**< log-posterior object */
        size_t mNumParams;/**< number of parameters in the posterior */
        realScalarType mMaxEps;/**< value of the epsilon of leapfrog */
        size_t mMaxNumSteps;/**< maximum number steps in the leapfrog */
        realVectorType const & mStartPoint;/**< staring point of the sampler */
        size_t mRandSeed;/**< random number seed */
        size_t mPacketSize;/**< number samples to be geerated in one go of HMC */
        size_t mNumBurn;/**< number of samples to be burned */
        size_t mNumSamples;/**< number of samples to be taken after burning */
        std::string mRootPathStr;/**< path-prefix to output files */
        bool mConsoleOutput;/**< do we need output to console? */
        std::string const & mDelimiter;/**< delimiter of the chain file */
        unsigned int mPrecision;/**< precision of the cahin file */
        realVectorType const & mKeDiagMInv;/**< diagonal elements of the inverse mass matrix */
    };
}

#endif //MPP_CANONICAL_HAMILTONIAN_SAMPLER_HPP

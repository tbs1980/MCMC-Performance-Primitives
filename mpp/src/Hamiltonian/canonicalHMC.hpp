#ifndef MPP_CANONICALHMC_HPP
#define MPP_CANONICALHMC_HPP

namespace mpp{ namespace Hamiltonian{

    /**
     * \class canonicalHMC
     *
     * \brief A class for performing canonical Hamiltonian Monte Carlo
     *
     * \tparam _randVarGenType random variate generator type
     * \tparam _potEngType potential energy (log-posterior) type
     * \tparam _kinEngType kinetic energy type
     * \tparam _integrationPolicy Hamiltonian integration policy
     *
     * This class will take a set of samples (packet) using canonical Hamiltonian
     * Monte Carlo method. The samples and the corresponding log posterior values
     * are returned.
     */
    template<class _randVarGenType,class _potEngType,class _kinEngType,class _integrationPolicy>
    class canonicalHMC
    {
    public:

        /**
         * \typedef _potEngType potEngType
         * \brief defines the potential energy type
         */
        typedef _potEngType potEngType;

        /**
         * \typedef _kinEngType kinEngType
         * \brief defines the kinetic energy type
         */
        typedef _kinEngType kinEngType;

        /**
         * \typedef _integrationPolicy integrationPolicy
         * \brief defines the integration policy for Hamiltonian
         */
        typedef _integrationPolicy integrationPolicy;

        /**
         * \typedef _randVarGenType randVarGenType
         * \brief defines the random variate generator type
         */
        typedef _randVarGenType randVarGenType;

        /**
         * \typedef typename randVarGenType::seedType seedType
         * \brief defines type of the seed for the random number generator
         */
        typedef typename randVarGenType::seedType seedType;

        /**
         * \typedef typename potEngType::realScalarType realScalarType
         * \brief defines the floating point type
         */
        typedef typename potEngType::realScalarType realScalarType;

        /**
         * \typedef typename Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType
         * \brief defines the real floating point vector type
         */
        typedef typename Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType;

        /**
         * \typedef typename Eigen::Matrix<realScalarType, Eigen::Dynamic, Eigen::Dynamic> realMatrixType
         */
        typedef typename Eigen::Matrix<realScalarType, Eigen::Dynamic, Eigen::Dynamic> realMatrixType;

        typedef typename _kinEngType::realScalarType realScalarTypeTypeKE;
        typedef typename _randVarGenType::realScalarType realScalarTypeTypeRVGen;

        static_assert(std::is_same<realScalarType,realScalarTypeTypeKE>::value,
            "POTENTIAL ENERGY AND KINTETIC ENERGY SHOULD SHOULD HAVE THE SAME FLOATING POINT TYPE");
        static_assert(std::is_same<realScalarType,realScalarTypeTypeRVGen>::value,
            "RANDOM VARARIATE GENERATOR AND KINTETIC ENERGY SHOULD SHOULD HAVE THE SAME FLOATING POINT TYPE");

        static const size_t MAX_NUM_STEPS = 100;/**< maximum number of steps allowed in integration */


        /**
         * \brief A constructor for setting up the HMC
         *
         * \param maxEps maximum value of epsilogn in the (leapfrog/Euler) integration
         * \param maxNumSteps maximum number of steps in the (leapfrog/Euler) integration
         * \param startPoint start-point of the Monte Carlo
         * \param rvGen random variate generator
         * \param G potentail energy or the log-posterior
         * \param K kinetic energy
         */
        canonicalHMC(realScalarType const maxEps,
            size_t const maxNumSteps,
            realVectorType const & startPoint,
            //seedType const seed,
            randVarGenType & rvGen,
            potEngType & G,
            kinEngType & K)
        :m_maxEps(maxEps),m_maxNumSteps(maxNumSteps),m_q0(startPoint),
        mRVGen(rvGen),m_G(G),m_K(K),m_accRate(0),mB(1),mLPVal(1),
        mProposal(startPoint.rows()),mPropLPVal(1)
        {
            BOOST_ASSERT_MSG(maxEps>0 and maxEps <2,"For stability of the leapfrog, we require 0<eps<2");
            BOOST_ASSERT_MSG(maxNumSteps>0 and maxNumSteps < MAX_NUM_STEPS,
                "We require 0<maxNumSteps<MAX_NUM_STEPS. You may edit MAX_NUM_STEPS at compile time");
            BOOST_ASSERT_MSG(K.numDims() == G.numDims(),
                "potentail and kinetic enegeries should have the same number of dimensions");
        }

        /**
         * \breif A function for generating samples.
         * \param samples     a real-matrix with dimensions num-samples x num-params
         * \param logPostVals a real-vector with diemsions num-samples
         *
         * This function returns the number of samples reqested along with the
         * corresponding log-posterior values. The \a samples should have the dimensionality
         * of num-samples x num-params and the \a logPostVals should have the dimensionality
         * of num-samples. Thus the nunber of samples to be generated can be controlled
         * by the dimensionality of \a samples ( and \a logPostVals)
         */
        void generate(realMatrixType & samples,realVectorType & logPostVals)
        {
            BOOST_ASSERT_MSG(samples.rows() == logPostVals.rows(),
                "number of samples should be equal to size of logPostVals");
            BOOST_ASSERT_MSG(samples.cols() == m_q0.rows(),
                "number of parameters in the samples does not match the number of parameters in the start point");
            size_t numSamples = (size_t) samples.rows();
            size_t numParams = (size_t) samples.cols();

            size_t iter = 0;
            size_t samp = 0;

            // loop over and generate requested number of samples
            while(samp < numSamples)
            {
                // 1) propose a new (q,p) tuple
                realVectorType q0(m_q0);

                // 2) randomise the trajectory
                realScalarType u = mRVGen.uniform();
                const realScalarType eps = m_maxEps*u;
                u = mRVGen.uniform();
                const size_t numSteps = (size_t)(u*(realScalarType)m_maxNumSteps);

                // 3) generate a random momentum vector
                realVectorType p0(numParams);
                for(size_t i=0;i<numParams;++i)
                {
                    p0(i) = mRVGen.normal();
                }

                // 4) find the Hamiltonian at (q0,p0)
                m_K.rotate(p0);
                realScalarType valG(0);
                realScalarType valK(0);
                m_G.value(q0,valG); //TODO I have already calculated this in the previous interaion
                m_K.value(p0,valK);
                realScalarType h0 = -(valG+valK);

                // 5) integrate the phase space using discretisation
                integrationPolicy::integrate(eps,numSteps,m_G,m_K,q0,p0);

                // 6) find the Hamiltonian again
                m_G.value(q0,valG);
                m_K.value(p0,valK);
                realScalarType h1 = -(valG+valK);

                // 7) store the current state of the MCMC
                mProposal = q0;
                mPropLPVal = valG;
                mMHVal = h1;

                // 8) accept/reject
                realScalarType dH = h1-h0;
                u = mRVGen.uniform();
                if(std::log(u) < -dH*mB) // by default temperature is 1
                {
                    // 9) copy required stuff
                    m_q0 = q0;
                    samples.row(samp) = m_q0;
                    logPostVals(samp) = valG;
                    ++samp;
                }
                iter++;
            }
            m_accRate = (realScalarType)samp/(realScalarType)iter;
        }

        /**
         * \brief A function that retunes the current acceptance rate.
         * @return  current acceptance rate
         */
        inline realScalarType getAcceptanceRate(void) const
        {
            return m_accRate;
        }

        /**
         * \brief A function for setting the seed of the random number generator
         * @param seed seed of the random number generator
         */
        inline void setSeed(seedType const seed)
        {
            mSeed = seed;
            mRVGen.seed(seed);
        }

        /**
         * \brief A function that returns current state of the random number generator
         * @return  current state of the random number generator
         */
        inline std::string getRandState(void) const
        {
            std::stringstream state;
            mRVGen.getState(state);
            return state.str();
        }

        /**
         * \brief A function that sets current state of the random number generator
         * @param stateStr state of the random number generator to be set
         */
        inline void setRandState(std::string & stateStr)
        {
            std::stringstream state;
            state<<stateStr;
            mRVGen.setState(state);
        }

        /**
         * \brief A function that returns the start-point of the HMC
         * @return  the start-point of the HMC
         */
        inline realVectorType getStartPoint(void) const
        {
            return m_q0;
        }

        /**
         * \brief A function that returns the start-point of the HMC
         * @param q0 the start-point of the HMC to be set
         */
        inline void setStartPoint(realVectorType const & q0)
        {
            BOOST_ASSERT_MSG(q0.rows() == m_q0.rows(),
            "Dimensions of input q0 does not agree with the that of m_q0");
            m_q0 = q0;
        }

        /**
         * \brief A function that returns the number of parameters in the posterior
         * @return the number of parameters in the posterior
         */
        inline size_t numParams(void) const
        {
            return (size_t) m_q0.rows();
        }

        /**
         * \brief A function that returns the tempering temperature of the chain
         * @return  the tempering temperature of the chain
         */
        inline realScalarType getTempB(void) const
        {
            return mB;
        }

        /**
         * \brief A function that sets the tempering temperature of the chain
         * @param  the tempering temperature of the chain
         */
        inline void setTempB(realScalarType const B)
        {
            BOOST_ASSERT_MSG(B>=0 and B<=1, "The parameter B should be between 0 and 1");
            mB = B;
        }

        /**
         * \brief A function that returns the value used for Metropolis-Hastings crieterion
         * @return  the value used for Metropolis-Hastings crieterion
         */
        inline realScalarType getMHVal(void) const
        {
            return mMHVal;
        }

        /**
         * \brief A function that returns the log-posterior value
         * @return  the log-posterior value
         */
        inline realScalarType getLogPostVal(void) const
        {
            return mLPVal;
        }

        /**
         * \brief A function that sets the log-posterior value
         * @param logPostVal the log-posterior value
         */
        inline void setLogPostVal(realScalarType logPostVal)
        {
            mLPVal = logPostVal;
        }

        /**
         * \brief A function that returns the HMC proposal
         * @return  the HMC proposal
         */
        inline realVectorType getProposal(void) const
        {
            return mProposal;
        }

        /**
         * \brief A function that sets the
         * @param proposal [description]
         */
        inline void setProposal(realVectorType const & proposal)
        {
            BOOST_ASSERT_MSG(mProposal.rows() == mProposal.rows(),
            "Dimensions of input q0 does not agree with the that of m_q0");
            mProposal = proposal;
        }

        /**
         * \brief A function that returns the log-posterior value of the proposal
         * @return  the log-posterior value of the proposal
         */
        inline realScalarType getPropLPVal(void) const
        {
            return mPropLPVal;
        }

        /**
         * \breif A function that returns the seed of the random number generator
         * @return  the seed of the random number generator
         */
        inline seedType getSeed(void) const
        {
            return mSeed;
        }

        /**
         * \brief A function that returns the address of the random variate generator
         * @return the address of the random variate generator
         */
        inline randVarGenType & getRVGen() const
        {
            return mRVGen;
        }

    private:
        realScalarType m_maxEps; /**< maximum value of epsilon */
        size_t m_maxNumSteps;/**< maximum number of steps */
        realVectorType m_q0;/**< start-point */
        seedType mSeed; /**< random number generator seed  */
        randVarGenType & mRVGen; /**< random variate generator */
        potEngType & m_G; /**< potential energy */
        kinEngType & m_K; /**< kinetic energy */
        realScalarType m_accRate; /**< acceptance rate */

        realScalarType mB; /**< tempering temperature */
        realScalarType mMHVal; /**< Metropolis-Hastings value */
        realScalarType mLPVal; /**< log-posterior value */
        realVectorType mProposal; /**< HMC proposal */
        realScalarType mPropLPVal; /**< log-posterior value of the proposal  */
    };

}//Hamiltonian
}//namespace mpp

#endif //MPP_CANONICALHMC_HPP

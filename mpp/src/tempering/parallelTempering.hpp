#ifndef MPP_PARALLEL_TEMPERING_HPP
#define MPP_PARALLEL_TEMPERING_HPP

namespace mpp{ namespace prltemp {

    /**
     * \class powerLawTemperature
     *
     * \tparam _realScalarType real floating point type
     *
     * \brief A class for computing power law based B-coefficients for tempering.
     *
     * This class computes the B-coefficients for the parallel tempering using a power
     * law, B = fact^(i/numChains).
     */
    template<typename _realScalarType>
    class powerLawTemperature
    {
    public:

        /**
         * \typedef _realScalarType realScalarType
         * \brief defines the real floating point type
         */
        typedef _realScalarType realScalarType;

        static const size_t NUM_MCMC_CHAINS=100; /**< maximum number of mcmc chains in tempering */

        /**
         * \brief A constructor that sets up the power law based B coefficents
         *
         * \param fact base of the power law
         * \param numChains number of chains in the parallel tempering
         */
        powerLawTemperature(realScalarType const fact,size_t const numChains)
        :mFact(fact),mNumChains(numChains)
        {
            BOOST_ASSERT_MSG(fact>0 and fact<1,
                "The value of power-factor should between 0 and 1.");
            BOOST_ASSERT_MSG(numChains<=NUM_MCMC_CHAINS,
                "For safety a maximum value for number states is set here. Re-compile with higher values.");
        }

        /**
         * \brief This function returns the value of the B-coefficient
         * @param  i chain number (first chains is 0)
         * @return   the value of the B-coefficient
         */
        inline realScalarType value(size_t i) const
        {
            BOOST_ASSERT(i>=0 and i<mNumChains);
            realScalarType temp = (realScalarType)i / (realScalarType)mNumChains;
            return std::pow(mFact,temp);
        }

        /**
         * \brief A function that returns the number of chains in the parallel tempering
         * @return  the number of chains in the parallel tempering
         */
        inline size_t numChains(void) const
        {
            return mNumChains;
        }

    private:
        realScalarType mFact; /**< base of the power law*/
        size_t mNumChains; /**< number of chains in the parallel tempering */
    };

    /**
     * \class parallelTemperingMCMC
     *
     * \brief This class performs parallel tempering on the MCMC provided
     *
     * \tparam _MCMCType type of MCMC in the parallel tempering
     * \tparam _chainTempType type of the B-coefficients
     *
     * This class performs parallel tempering on the MCMC using a set of chain
     * temperatures computed using the method provided.
     */
    template<class _MCMCType,class _chainTempType>
    class parallelTemperingMCMC
    {
    public:

        /**
         * \typedef _MCMCType MCMCType
         * \brief defines the MCMC type
         */
        typedef _MCMCType MCMCType;

        /**
         * \typedef _chainTempType chainTempType
         * \brief defines the method by which the chain temperatures are computed
         */
        typedef _chainTempType chainTempType;

        /**
         * \typedef typename MCMCType::realScalarType realScalarType
         * \brief real floating point type
         */
        typedef typename MCMCType::realScalarType realScalarType;

        typedef typename chainTempType::realScalarType realScalarTypeChain;

        /**
         * \typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType
         * \brief real floating point vector type
         */
        typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType;

        /**
         * \typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, Eigen::Dynamic> realMatrixType
         * \brief real floating point matrix type
         */
        typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, Eigen::Dynamic> realMatrixType;

        /**
         * \typedef typename MCMCType::randVarGenType randVarGenType
         * \brief random variate generator type
         */
        typedef typename MCMCType::randVarGenType randVarGenType;

        /**
         * \typedef typename realVectorType::Index indexType
         * \brief integer type
         */
        typedef typename realVectorType::Index indexType;

        static_assert(std::is_same<realScalarType,realScalarTypeChain>::value,
            "MCMC AND CHAIN-TEMPERATURE SHOULD SHOULD HAVE THE SAME FLOATING POINT TYPE");

        static const size_t MAX_NUM_STATES = 100; /**< maximum number of chains in tempering */

        /**
         * \brief A constructor that sets up the parallel tempering
         *
         * \param MCMC a vector of MCMCs
         * \param swapRatio the ratio with which swaps are performed. 0 for none 1 for all.
         * \param chainTemps a method for computing the chain temperatures
         */
        parallelTemperingMCMC(std::vector<MCMCType> & MCMC,
            realScalarType const swapRatio,
            chainTempType const & chainTemps)
        :mB(MCMC.size()),mMCMC(MCMC),mSwapRatio(swapRatio),
        mChainTemps(chainTemps),mRVGen(MCMC[0].getRVGen()),mAccRate(1)
        {
            BOOST_ASSERT_MSG(MCMC.size() <= MAX_NUM_STATES,
                "For safety a maximum value for number states is set here. Re-compile with higher values.");
            BOOST_ASSERT_MSG(swapRatio>0 and swapRatio<=1,
                "Swap-ratio should be a real number btween 0 and 1.");
            BOOST_ASSERT_MSG(mMCMC.size() == mChainTemps.numChains(),
                "Number of chains in MCMC does not match the number of chains in chain-temperatures.");

            for(size_t i=0;i<mMCMC.size();++i)
            {
                mB(i) = chainTemps.value(i);
                mMCMC[i].setTempB(mB(i));
            }

            mNumParams = mMCMC[0].numParams();
            mNumChains = mMCMC.size();
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
            BOOST_ASSERT_MSG((size_t) samples.cols() == mNumParams,
                "number of parameters in the samples does not match the number of parameters in the start point");
            BOOST_ASSERT_MSG(samples.rows() % (indexType) mNumChains == 0,
                "The number of rows in samples should be an integer multiple of number of chains in tempering.");

            size_t numSamples = (size_t) samples.rows() / mNumChains;
            size_t mNumChains = mMCMC.size();

            realScalarType accRate = 0.;

            for(size_t i=0;i<numSamples;++i)
            {
                realMatrixType singleSample(1,mNumParams);
                realVectorType singleLogPostVal(1);

                // 1) try to generate a single sample from all MCMCs
                mMCMC[0].generate(singleSample,singleLogPostVal);
                accRate += mMCMC[0].getAcceptanceRate();
                for(size_t j=1;j<mNumChains;++j)
                {
                    mMCMC[j].generate(singleSample,singleLogPostVal);
                }

                // 2) swap states
                for(size_t j=0;j<mNumChains;++j)
                {
                    // 3) generate a uniform random number
                    realScalarType u = mRVGen.uniform();

                    // 4) are we swapping?
                    if(u < mSwapRatio)
                    {
                        if(j+1 < mNumChains)
                        {
                            // 5) compute the swapping probability alpha
                            realScalarType logAlpha = (mB[j]-mB[j+1])
                                *(mMCMC[j+1].getMHVal()-mMCMC[j].getMHVal());

                            // 6) accept/reject
                            u = mRVGen.uniform();
                            if(std::log(u) < logAlpha)
                            {
                                // 7) swap states
                                realVectorType qj =  mMCMC[j].getProposal();
                                realVectorType qj1 = mMCMC[j+1].getProposal();

                                mMCMC[j].setStartPoint(qj1);
                                mMCMC[j+1].setStartPoint(qj);

                                // 8) swap log posterior values
                                realScalarType lpj = mMCMC[j].getPropLPVal();
                                realScalarType lpj1 = mMCMC[j+1].getPropLPVal();

                                mMCMC[j].setLogPostVal(lpj);
                                mMCMC[j+1].setLogPostVal(lpj1);
                            }
                        }
                    }
                }

                // 9) copy the mcmc states to corresponding chains
                for(size_t j=0;j<mMCMC.size();++j)
                {
                    samples.row(j*numSamples+i) = mMCMC[j].getStartPoint();
                    logPostVals(j*numSamples+i) = mMCMC[j].getLogPostVal();
                }
            }

            // 10) find the mean acceptance rate
            mAccRate = accRate/(realScalarType)numSamples;

        }

        /**
         * \brief A function that retunes the current acceptance rate (of chain 0).
         * @return  current acceptance rate (of chain 0)
         */
        inline realScalarType getAcceptanceRate(void) const
        {
            return mAccRate;
        }

        /**
         * \brief A function that returns the number of parameters in the log-posterior
         * @return  the number of parameters in the log-posterior
         */
        inline size_t numParams(void) const
        {
            return mNumParams;
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
         * \brief A function that returns the address of the random variate generator
         * @return the address of the random variate generator
         */
        inline randVarGenType & getRVGen() const
        {
            return mRVGen;
        }

        /**
         * \brief A function that returns the start-point of the HMC
         * @param q0 the start-point of the HMC to be set
         */
        inline void setStartPoint(realVectorType const & q0)
        {
            //std::cout<<q0.rows()<<"\t"<<mNumParams<<"\t"<<mNumChains<<std::endl;
            BOOST_ASSERT_MSG((size_t)q0.rows() == mNumParams*mNumChains ,
            "Dimensions of input q0 does not agree with mNumParams*mNumChains");

            for(size_t i=0;i<mNumChains;++i)
            {
                realVectorType stp = q0.segment(i*mNumParams,mNumParams);
                //std::cout<<"for chain "<<i<<" is "<<q0.transpose()<<std::endl;
                mMCMC[i].setStartPoint(stp);
            }
        }

    private:
        realVectorType mB; /**< B-coefficients */
        std::vector<MCMCType> & mMCMC; /**< MCMCs */
        realScalarType mSwapRatio; /**< swap ratio for parallel chains */
        chainTempType mChainTemps; /**<  chain temperature calculator */
        randVarGenType & mRVGen; /**< random variate generator */
        realScalarType mAccRate; /**< acceptance rate of chain 0  */
        //std::vector<realMatrixType> mSamples; /**< samples from all chains */
        //std::vector<realVectorType> mLogPostVals; /**< log-posterior values from all chains */
        realMatrixType mSamples;
        realVectorType mLogPostVals;
        size_t mNumParams; /**< number of parameters in the log-posterior */
        size_t mNumChains; /**< number of chains in the parallel tempering */
    };

}//namespace pt
}//namespace mpp

#endif //MPP_PARALLEL_TEMPERING_HPP

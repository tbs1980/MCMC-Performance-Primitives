#ifndef MPP_PARALLEL_TEMPERING_HPP
#define MPP_PARALLEL_TEMPERING_HPP

namespace mpp{ namespace prltemp {

    template<class _MCMCType>
    class parallelTemperingMCMC
    {
    public:

        typedef _MCMCType MCMCType;
        typedef typename MCMCType::realScalarType realScalarType;
        typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType;
        typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, Eigen::Dynamic> realMatrixType;
        typedef typename MCMCType::randVarGenType randVarGenType;
        typedef typename realVectorType::Index indexType;

        static const size_t MAX_NUM_STATES = 100;

        parallelTemperingMCMC(std::vector<MCMCType> & MCMC,realScalarType const swapRatio)
        :mB(MCMC.size()),mMCMC(MCMC),mSwapRatio(swapRatio),mRVGen(0),mAccRate(1)
        {
            BOOST_ASSERT_MSG(MCMC.size() <= MAX_NUM_STATES,
                "For safety a maximum value for number states is set here. Re-compile with higher values.");
            BOOST_ASSERT_MSG(swapRatio>0 and swapRatio<=1,
                "Swap-ratio should be a real number btween 0 and 1.");
            // this can be a policy
            for(size_t i=0;i<mMCMC.size();++i)
            {
                realScalarType temp = (realScalarType)(i)/(mMCMC.size());
                mB(i) = std::pow(0.005,temp);
                mMCMC[i].setTempB(mB(i));
                mMCMC[i].setSeed(i);//TODO change this
            }

            mNumParams = mMCMC[0].numParams();
            mNumChains = mMCMC.size();
        }

        void generate(realMatrixType & samples,realVectorType & logPostVals)
        {
            BOOST_ASSERT_MSG(samples.rows() == logPostVals.rows(),
                "number of samples should be equal to size of logPostVals");
            BOOST_ASSERT_MSG(samples.cols() == mNumParams,
                "number of parameters in the samples does not match the number of parameters in the start point");

            size_t numSamples = (size_t) samples.rows();
            size_t mNumChains = mMCMC.size();

            mSamples = std::vector<realMatrixType>(mNumChains,samples);
            mLogPostVals = std::vector<realVectorType>(mNumChains,logPostVals);

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
                    mSamples[j].row(i) = mMCMC[j].getStartPoint();
                    mLogPostVals[j](i) = mMCMC[j].getLogPostVal();
                }
            }

            // 10) find the mean acceptance rate
            mAccRate = accRate/(realScalarType)numSamples;

            // 11) finally copy temp=1 chains for return
            samples = mSamples[0];
            logPostVals = mLogPostVals[0];

        }

        inline realScalarType getAcceptanceRate(void) const
        {
            return mAccRate;
        }

        inline realMatrixType getSamplesFromChain(size_t i) const
        {
            BOOST_ASSERT_MSG(mSamples.size()>0,
                "The samples has not been computed yet. Please generate them first.");
            BOOST_ASSERT(i >=0 and i<mNumChains);
            return mSamples[i];
        }

        inline realVectorType getLogPostValsFromChain(size_t i) const
        {
            BOOST_ASSERT_MSG(mLogPostVals.size()>0,
                "The samples has not been computed yet. Please generate them first.");
            BOOST_ASSERT(i >=0 and i<mNumChains);
            return mLogPostVals[i];
        }

        inline size_t numParams(void) const
        {
            return mNumParams;
        }

    private:
        realVectorType mB;
        std::vector<MCMCType> & mMCMC;
        realScalarType mSwapRatio;
        randVarGenType mRVGen;
        realScalarType mAccRate;
        std::vector<realMatrixType> mSamples;
        std::vector<realVectorType> mLogPostVals;
        size_t mNumParams;
        size_t mNumChains;
    };

}//namespace pt
}//namespace mpp

#endif //MPP_PARALLEL_TEMPERING_HPP

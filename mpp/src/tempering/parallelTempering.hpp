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

        parallelTemperingMCMC(std::vector<MCMCType> & MCMC)
        :mB(MCMC.size()),mMCMC(MCMC),mRVGen(0),mAccRate(1)
        {
            BOOST_ASSERT_MSG(MCMC.size() <= MAX_NUM_STATES,
                "For safety a maximum value for number states is set here. Re-compile with higher values.");
            // this can be a policy
            for(size_t i=0;i<mMCMC.size();++i)
            {
                realScalarType temp = (realScalarType)(i)/(mMCMC.size());
                mB(i) = std::pow(0.005,temp);
                mMCMC[i].setTempB(mB(i));
                mMCMC[i].setSeed(i);//TODO change this
            }
        }

        void generate(realMatrixType & samples,realVectorType & logPostVals)
        {
            size_t numSamples = (size_t) samples.rows();

            size_t iter = 0;
            size_t samp = 0;

            while(samp < numSamples)
            {
                // 1) try to generate a single sample from all MCMCs
                bool accepted =  mMCMC[0].try2Generate();
                for(size_t j=1;j<mMCMC.size();++j)
                {
                    mMCMC[j].try2Generate();
                }

                // 2) swap states
                for(size_t j=0;j<mMCMC.size();++j)
                {
                    // 3) generate a uniform random number
                    realScalarType u = mRVGen.uniform();

                    // 4) are we swapping?
                    if(u < 0.5)
                    {
                        if(j+1 < mMCMC.size())
                        {
                            // 5) compute the swapping probability alpha
                            realScalarType lalpha = (mB[j]-mB[j+1])
                                *(mMCMC[j+1].getMHVal()-mMCMC[j].getMHVal());

                            // 6) accept/reject
                            u = mRVGen.uniform();
                            if(std::log(u) < lalpha)
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

                // finally copy the values form B=1
                if(accepted)
                {
                    samples.row(samp) = mMCMC[0].getStartPoint();
                    logPostVals(samp) = mMCMC[0].getLogPostVal();
                    ++samp;
                }

                ++iter;
            }
            mAccRate = (realScalarType)samp/(realScalarType)iter;

        }

        inline realScalarType getAcceptanceRate(void) const
        {
            return mAccRate;
        }

    private:
        realVectorType mB;
        std::vector<MCMCType> & mMCMC;
        randVarGenType mRVGen;
        realScalarType mAccRate;
    };

}//namespace pt
}//namespace mpp

#endif //MPP_PARALLEL_TEMPERING_HPP

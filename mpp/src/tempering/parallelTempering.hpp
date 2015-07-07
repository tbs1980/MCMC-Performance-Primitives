#ifndef MPP_PARALLEL_TEMPERING_HPP
#define MPP_PARALLEL_TEMPERING_HPP

namespace mpp{ namespace prltemp {

    template<class _MCMCType>
    class parallelTemperingMCMC
    {
    public:

        typedef _MCMCType MCMCType;
        typedef typename MCMCType::realScalarType;
        typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType;
        typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, Eigen::Dynamic> realMatrixType;
        typedef typename MCMCType::randVarGenType randVarGenType;
        typedef typename realVectorType::Index indexType;

        static const size_t MAX_NUM_STATES = 100;

        parallelTemperingMCMC(std::vector<MCMCType> & MCMC)
        :mB(MCMC.size()),mMCMC(MCMC)
        {
            BOOST_ASSERT_MSG(numStates <= MAX_NUM_STATES,
                "For safety a maximum value for number states is set here. Re-compile with higher values.");
            // this can be a policy
            for(size_t i=0;i<mMCMC.size();++i)
            {
                temp = (realScalarType)(i)/(mMCMC.size()-1);
                mB(i) = std::pow(0.005,temp);
                mMCMC[i].setTempB(mB(i));
            }
        }

        void generate(realMatrixType & samples,realVectorType & logPostVals)
        {
            for(indexType i=0;i<samples.rows();++i)
            {
                // generate a single sample from all MCMCs
                for(size_t j=0;j<mMCMC.size();++j)
                {
                    realMatrixType singleSample(1,samples.cols());
                    realVectorType singleLPVal(1);

                    mMCMC[i].generate(singleSample,singleLPVal);
                }

                for(size_t j=0;j<mMCMC.size();++j)
                {
                    // generate a uniform random number
                    realScalarType u = m_rVGen.uniform();

                    if(u < 0.5)
                    {
                        if(j+1 < mMCMC.size())
                        {
                            realScalarType lalpha = (mB[j]-mB[j+1])
                                *(mMCMC[j+1].getMHVal()-mMCMC[j].getMHVal());
                            u = m_rVGen.uniform();
                            if(std::log(u) < lalpha)
                            {
                                // swap states
                                realVectorType qj =  mMCMC[j].getStartPoint();
                                realVectorType qj1 = mMCMC[j+1].getStartPoint();

                                mMCMC[j].setStartPoint(qj1);
                                mMCMC[j+1].setStartPoint(qj);

                                // swap log posterior values
                                realScalarType lpj = mMCMC[j].getLogPostVal();
                                realScalarType lpj1 = mMCMC[j+1].getLogPostVal();

                                mMCMC[j].setLogPostVal(lpj);
                                mMCMC[j+1].setLogPostVal(lpj1);
                            }
                        }
                    }
                }

                // finally copy the values form B=1
                samples.row(i) = mMCMC[0].getStartPoint();
                logPostVals(i) = mMCMC[0].getLogPostVal();
            }
        }

    private:
        realVectorType mB;
        std::vector<MCMCType> & mMCMC;
        randVarGenType m_rVGen;
    };

}//namespace pt
}//namespace mpp

#endif //MPP_PARALLEL_TEMPERING_HPP

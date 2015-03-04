#ifndef MPP_CANONICALHMC_HPP
#define MPP_CANONICALHMC_HPP

namespace mpp{ namespace Hamiltonian{

    /**
     * This class will take a set of samples (packet)
     */
    template<class _randVarGenType,class _potEngType,class _kinEngType,class _integrationPolicy>
    class canonicalHMC
    {
    public:

        typedef _potEngType potEngType;
        typedef _kinEngType kinEngType;
        typedef _integrationPolicy integrationPolicy;
        typedef _randVarGenType randVarGenType;

        typedef typename randVarGenType::seedType seedType;

        typedef typename potEngType::realScalarType realScalarType;
        typedef typename Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType;
        typedef typename Eigen::Matrix<realScalarType, Eigen::Dynamic, Eigen::Dynamic> realMatrixType;

        typedef typename _kinEngType::realScalarType realScalarTypeTypeKE;
        typedef typename _randVarGenType::realScalarType realScalarTypeTypeRVGen;

        static_assert(std::is_same<realScalarType,realScalarTypeTypeKE>::value,
            "POTENTIAL ENERGY AND KINTETIC ENERGY SHOULD SHOULD HAVE THE SAME FLOATING POINT TYPE");
        static_assert(std::is_same<realScalarType,realScalarTypeTypeRVGen>::value,
            "RANDOM VARARIATE GENERATOR AND KINTETIC ENERGY SHOULD SHOULD HAVE THE SAME FLOATING POINT TYPE");

        static const size_t MAX_NUM_STEPS = 100;


        canonicalHMC(realScalarType const maxEps,size_t const maxNumSteps,
            realVectorType const & startPoint,seedType const seed,
            potEngType & G,kinEngType & K)
        :m_maxEps(maxEps),m_maxNumSteps(maxNumSteps),m_q0(startPoint),
        m_rVGen(seed),m_G(G),m_K(K),m_accRate(0)
        {
            BOOST_ASSERT_MSG(maxEps>0 and maxEps <2,"For stability of the leapfrog, we require 0<eps<2");
            BOOST_ASSERT_MSG(maxNumSteps>0 and maxNumSteps < MAX_NUM_STEPS,
                "We require 0<maxNumSteps<MAX_NUM_STEPS. You may edit MAX_NUM_STEPS at compile time");
            BOOST_ASSERT_MSG(K.numDims() == G.numDims(),
                "potentail and kinetic enegeries should have the same number of dimensions");
        }

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
                // propose a new (q,p) tuple
                realVectorType q0(m_q0);

                // randomise the trajectory
                realScalarType u = m_rVGen.uniform();
                const realScalarType eps = m_maxEps*u;
                u = m_rVGen.uniform();
                const size_t numSteps = (size_t)(u*(realScalarType)m_maxNumSteps);

                // generate a random momentum vector
                realVectorType p0(numParams);
                for(size_t i=0;i<numParams;++i)
                {
                    p0(i) = m_rVGen.normal();
                }

                // find the Hamiltonian at (q0,p0)
                m_K.rotate(p0);
                realScalarType valG(0);
                realScalarType valK(0);
                m_G.value(q0,valG);
                m_K.value(p0,valK);
                realScalarType h0 = -(valG+valK);

                // integrate the phase space using discretisation
                integrationPolicy::integrate(eps,numSteps,m_G,m_K,q0,p0);
                //leapfrogIntegratorPolicy::integrate<potEngType,kinEngType>(eps,nSteps,G,K,q,p);

                // find the Hamiltonian again
                m_G.value(q0,valG);
                m_K.value(p0,valK);
                realScalarType h1 = -(valG+valK);

                // accept/reject
                realScalarType dH = h1-h0;
                u = m_rVGen.uniform();
                if(u < exp(-dH))
                {
                    // copy required stuff
                    m_q0=q0;
                    samples.row(samp) = m_q0;
                    logPostVals(samp) = valG;
                    ++samp;
                }
                iter++;
            }
            m_accRate = (realScalarType)samp/(realScalarType)iter;
        }

        inline realScalarType getAcceptanceRate(void) const
        {
            return m_accRate;
        }

        inline void setSeed(seedType const seed)
        {
            m_rVGen.seed(seed);
        }

        inline std::stringstream getRandState(void) const
        {
            std::stringstream state;
            m_rVGen.getState(state);
            return state;
        }

        inline void setRandState(std::stringstream & state)
        {
            m_rVGen.setState(state);
        }

        inline realVectorType getStartPoint(void) const
        {
            return m_q0;
        }

        inline void setStartPoint(realVectorType const & q0)
        {
            BOOST_ASSERT_MSG(q0.rows() == m_q0.rows(),
            "Dimensions of input q0 does not agree with the that of m_q0");
            m_q0 = q0;
        }

        inline size_t numParams(void) const
        {
            return (size_t) m_q0.rows();
        }

    private:
        realScalarType m_maxEps;
        size_t m_maxNumSteps;
        realVectorType m_q0;
        randVarGenType m_rVGen;
        potEngType & m_G;
        kinEngType & m_K;
        realScalarType m_accRate;
    };

}//Hamiltonian
}//namespace mpp

#endif //MPP_CANONICALHMC_HPP

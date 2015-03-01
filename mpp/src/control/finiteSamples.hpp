#ifndef MPP_FINITESAMPLES_HPP
#define MPP_FINITESAMPLES_HPP

namespace mpp { namespace control {

    template<typename _realScalarType>
    class finiteSamplesController
    {
    public:
        typedef typename Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType;
        typedef typename Eigen::Matrix<realScalarType, Eigen::Dynamic, Eigen::Dynamic> realMatrixType;

        static_assert(std::is_floating_point<realScalarType>::value,
            "PARAMETER SHOULD BE A FLOATING POINT TYPE");

        finiteSamplesController(size_t const numParams, size_t const packetSize,
            size_t const numBurn, size_t const numSamples,
            std::string const& rootPathStr, bool const consoleOutput, bool resume)
        :m_numParams(numParams), m_packetSize(), m_numSamples(numSamples),
            m_numBurn(numBurn), m_rootPathStr(rootPathStr),
            m_consoleOutput(consoleOutput), m_resume(resume), samplesTaken(0),
            samplesBurned(0), m_continueSampling(true)
        {

        }

        void dump(realMatrixType const & samples,std::stringstream const& randState,
            realScalarType const accRate)
        {
            size_t n = (size_t) samples.rows();

            // have we finished burning samples
            if(samplesBurned >= m_numBurn)
            {
                m_samplesTaken += n;
            }
            else
            {
                m_samplesBurned += n;
            }

            // should we continue sampling ?
            m_continueSampling = m_samplesTaken >= m_numSamples ? false : true;

            // dump the information
            if(resume)
            {
                // write the resume information to a file
            }

            // save the states
            m_startPoint = samples.row(n-1);
            m_accRate = accRate;
            m_randState = randState;
        }
    private:

        size_t m_numParams;
        size_t m_packetSize;
        size_t m_numBurn;
        size_t m_numSamples;
        std::string m_rootPathStr;
        bool m_consoleOutput;
        bool m_resume;

        size_t m_samplesTaken;
        size_t m_samplesBurned;

        bool m_continueSampling;

        realVectorType m_startPoint;
        realScalarType m_accRate;
        std::string m_randState;
    }

}//namespace control
}//namespace mpp

#endif //MPP_FINITESAMPLES_HPP

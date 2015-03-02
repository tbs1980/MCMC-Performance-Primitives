#ifndef MPP_FINITESAMPLES_HPP
#define MPP_FINITESAMPLES_HPP

namespace mpp { namespace control {

    template<typename _realScalarType>
    class finiteSamplesControl
    {
        friend class boost::serialization::access;
    public:
        typedef _realScalarType realScalarType;
        typedef typename Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType;
        typedef typename Eigen::Matrix<realScalarType, Eigen::Dynamic, Eigen::Dynamic> realMatrixType;

        static_assert(std::is_floating_point<realScalarType>::value,
            "PARAMETER SHOULD BE A FLOATING POINT TYPE");

        finiteSamplesControl(size_t const numParams, size_t const packetSize,
            size_t const numBurn, size_t const numSamples,
            std::string const& rootPathStr, bool const consoleOutput,
            realVectorType const & startPoint, std::string const& randState)
        :m_numParams(numParams), m_packetSize(packetSize),m_numBurn(numBurn),
            m_numSamples(numSamples), m_rootPathStr(rootPathStr),
            m_consoleOutput(consoleOutput), m_samplesTaken(0),
            m_samplesBurned(0), m_continueSampling(true),m_startPoint(numParams),
            m_randState(randState),m_accRate(0)
        {
            BOOST_ASSERT_MSG(numParams >0,"At least one parameters is required.");
            BOOST_ASSERT_MSG(packetSize>0,"Packet size should be greater than zero");
            BOOST_ASSERT_MSG(numParams == (size_t) startPoint.rows(),
                "The startpoint should have numParams number of rows");

            m_archiveOutFile = m_rootPathStr + std::string(".resume");
            for(size_t i=0;i<m_numParams;++i)
            {
                m_startPoint[i] = startPoint(i);
            }
        }

        void dump(realMatrixType const & samples,realScalarType const accRate,
            std::string const& randState)
        {
            BOOST_ASSERT_MSG((size_t) samples.rows() == m_packetSize ,
                "Number of rows in samples should be identical to the packet size");
            size_t n = (size_t) samples.rows();

            // have we finished burning samples
            if(m_samplesBurned >= m_numBurn)
            {
                m_samplesTaken += n;
            }
            else
            {
                m_samplesBurned += n;
            }

            // should we continue sampling ?
            m_continueSampling = m_samplesTaken >= m_numSamples ? false : true;

            // write the resume information to a file
            std::ofstream ofs(m_archiveOutFile);
            boost::archive::binary_oarchive oa(ofs);
            oa << (*this);

            // save the states
            for(size_t i=0;i<m_numParams;++i)
            {
                m_startPoint[i] = samples(n-1,i);
            }
            m_accRate = accRate;
            m_randState = randState;
        }
    private:

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & m_numParams;
            ar & m_packetSize;
            ar & m_numBurn;
            ar & m_numSamples;
            ar & m_rootPathStr;
            ar & m_consoleOutput;
            ar & m_samplesTaken;
            ar & m_samplesBurned;
            ar & m_continueSampling;
            ar & m_startPoint;
            //ar & m_accRate;
            ar & m_randState;

            if(version > 0)
            {
                ar & m_accRate;
            }
        }

        size_t m_numParams;
        size_t m_packetSize;
        size_t m_numBurn;
        size_t m_numSamples;
        std::string m_rootPathStr;
        bool m_consoleOutput;

        size_t m_samplesTaken;
        size_t m_samplesBurned;

        bool m_continueSampling;

        std::vector<realScalarType> m_startPoint;
        std::string m_randState;
        realScalarType m_accRate;

        std::string m_archiveOutFile;
    };

}//namespace control
}//namespace mpp

#endif //MPP_FINITESAMPLES_HPP

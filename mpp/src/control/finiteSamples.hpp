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

            m_archiveOutFileName = m_rootPathStr + std::string(".resume");
            m_chainOutFileName = m_rootPathStr + std::string(".chain");
            for(size_t i=0;i<m_numParams;++i)
            {
                m_startPoint[i] = startPoint(i);
            }

            //see if the resume file exists?
            if(boost::filesystem::exists(m_archiveOutFileName))
            {
                if(m_consoleOutput)
                {
                    BOOST_LOG_TRIVIAL(info) << "Resume file found. Starting sampling from the previous state.";
                }

                //now load the information from the resume file
                std::ifstream ifs(m_archiveOutFileName);
                boost::archive::binary_iarchive ia(ifs);
                ia >> (*this);
                ifs.close();
            }
            else
            {
                // we didn't find any resume files. Can we open one for writing
                std::ofstream ofs(m_archiveOutFileName);
                if(ofs.is_open())
                {
                    // yes. now close and remove that file
                    ofs.close();
                    boost::filesystem::remove(m_archiveOutFileName);
                }
                else
                {
                    // we cannnot open this flie for writing. stop
                    if(m_consoleOutput)
                    {
                        BOOST_LOG_TRIVIAL(error) << "Cannot open the resume file "<<m_archiveOutFileName<< "For writing";
                    }
                    // no need to proceed
                    m_continueSampling = false;
                }
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
                if(m_consoleOutput)
                {
                    BOOST_LOG_TRIVIAL(info) << "Number of samples taken = "<<m_samplesTaken;
                }
            }
            else
            {
                m_samplesBurned += n;
                if(m_consoleOutput)
                {
                    BOOST_LOG_TRIVIAL(info) << "Number of samples burned = "<<m_samplesBurned;
                }
            }

            // should we continue sampling ?
            m_continueSampling = m_samplesTaken >= m_numSamples ? false : true;

            // save the states
            for(size_t i=0;i<m_numParams;++i)
            {
                m_startPoint[i] = samples(n-1,i);
            }
            m_accRate = accRate;
            m_randState = randState;

            // write the resume information to a file
            std::ofstream ofs(m_archiveOutFileName);
            boost::archive::binary_oarchive oa(ofs);
            oa << (*this);
            ofs.close();

            //has the sampling finished?
            if(m_continueSampling == false)
            {
                if(m_consoleOutput)
                {
                    BOOST_LOG_TRIVIAL(info) << "Sampling finished";
                }
                // delete the archive file
                boost::filesystem::remove(m_archiveOutFileName);
                // rename the chain file
                renameChainFile();
            }
        }

        std::string const & getChainFileName() const
        {
            return m_chainOutFileName;
        }

        realVectorType getStartPoint() const
        {
            realVectorType q0(m_numParams);
            for(size_t i=0;i<m_numParams;++i)
            {
                q0(i) = m_startPoint[i];
            }
            return q0;
        }

        std::string const& getRandState() const
        {
            return m_randState;
        }

        size_t numParams() const
        {
            return m_numParams;
        }

        bool continueSampling() const
        {
            return m_continueSampling;
        }

        size_t packetSize() const
        {
            return m_packetSize;
        }

    private:

        void renameChainFile(void)
        {

            using namespace boost::local_time;
            time_zone_ptr zone(new posix_time_zone("MST-07"));
            local_date_time ldt = local_sec_clock::local_time(zone);
            std::stringstream ss;
            local_time_facet* output_facet = new local_time_facet();
            ss.imbue(std::locale(std::locale::classic(), output_facet));
            output_facet->format(local_time_facet::iso_time_format_specifier);
            ss << ldt;
            std::string finishedChainOutFileName = m_rootPathStr + std::string(".")
                + ss.str() + std::string(".finished.chain");

            if(m_consoleOutput)
            {
                BOOST_LOG_TRIVIAL(info) << "Renaming the chain file to "<<finishedChainOutFileName;
            }

            boost::filesystem::rename(m_chainOutFileName,finishedChainOutFileName);

        }

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

            if(version >= 0)
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

        std::string m_archiveOutFileName;
        std::string m_chainOutFileName;
    };

}//namespace control
}//namespace mpp

#endif //MPP_FINITESAMPLES_HPP

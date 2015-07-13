#ifndef MPP_FINITESAMPLES_HPP
#define MPP_FINITESAMPLES_HPP

namespace mpp { namespace control {

    /**
     * \class finiteSamplesControl
     *
     * \tparam _realScalarType real floating point type
     *
     * \brief A class for controlling the sampler
     *
     * This class controlls the behaviour of the sampler, i.e. it stops the sampling
     * process after a finite number samples are taken. It also dumps the state of
     * the MCMC to a resume file every a dump function is called. Once the required
     * number of sampels are generated, this class renames the chain file by prepending
     * it with a time-stamp.
     */
    template<typename _realScalarType>
    class finiteSamplesControl
    {
        /**
         * \brief serialization of the data
         * \relates  boost::serialization::access
         */
        friend class boost::serialization::access;
    public:

        /**
         * \typedef _realScalarType realScalarType
         * \brief real floating point type
         */
        typedef _realScalarType realScalarType;

        /**
         * \typedef typename Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType
         * \brief real floating point vector type
         */
        typedef typename Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType;

        /**
         * \typedef typename Eigen::Matrix<realScalarType, Eigen::Dynamic, Eigen::Dynamic> realMatrixType
         * \brief real floating point vector type
         */
        typedef typename Eigen::Matrix<realScalarType, Eigen::Dynamic, Eigen::Dynamic> realMatrixType;

        static_assert(std::is_floating_point<realScalarType>::value,
            "PARAMETER SHOULD BE A FLOATING POINT TYPE");

        /**
         * \brief A constructor that sets up the controller
         *
         * \param numParams number of parameters in the log-posterior
         * \param packetSize packet-size for sampling
         * \param numBurn number of samples to be burned
         * \param numSamples number of samples to taken after burning
         * \param rootPathStr path-prefix to output files
         * \param consoleOutput a flag for controlling output to console
         * \param startPoint a start point for MCMC chains
         * \param randState state of the random number generator
         */
        finiteSamplesControl(size_t const numParams,
            size_t const packetSize,
            size_t const numBurn,
            size_t const numSamples,
            std::string const& rootPathStr,
            bool const consoleOutput,
            realVectorType const & startPoint,
            std::string const& randState)
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

        /**
         * \brief A function that dumps the current state of MCMC
         * @param samples   current pack of samples
         * @param accRate   current acceptance rate
         * @param randState current random number generator state
         *
         * This function is called every packet-size iterations. In other words,
         * every time we have samples the packet of samples, this function is called
         * to dump information. It also checks for the number of samples taken, and if
         * the specified number of samples are genrated, a flag for stopping the
         * sampler is set. The current state of the sampler is written to the disk
         * ever time this function is callled.
         */
        void dump(realMatrixType const & samples,realScalarType const accRate,
            std::string const& randState)
        {
            BOOST_ASSERT_MSG((size_t) samples.rows() == m_packetSize ,
                "Number of rows in samples should be identical to the packet size");
            size_t n = (size_t) samples.rows();

            realScalarType precAccRate = accRate*100;

            // have we finished burning samples
            if(m_samplesBurned >= m_numBurn)
            {
                m_samplesTaken += n;
                if(m_consoleOutput)
                {
                    BOOST_LOG_TRIVIAL(info) << "Number of samples taken = "<<m_samplesTaken
                        <<", Acceptance = "<<precAccRate<< "%";
                }
            }
            else
            {
                m_samplesBurned += n;
                if(m_consoleOutput)
                {
                    BOOST_LOG_TRIVIAL(info) << "Number of samples burned = "<<m_samplesBurned
                        <<", Acceptance = "<<precAccRate<< "%";
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

        /**
         * \brief A function that returns the name of chain file.
         * @return the name of chain fil
         */
        std::string const & getChainFileName() const
        {
            return m_chainOutFileName;
        }

        /**
         * \brief A function that returns the start-point of the MCMC
         * @return the start-point of the MCMC
         */
        realVectorType getStartPoint() const
        {
            realVectorType q0(m_numParams);
            for(size_t i=0;i<m_numParams;++i)
            {
                q0(i) = m_startPoint[i];
            }
            return q0;
        }

        /**
         * \brief A function that retunrs the random number generator state
         * @return the random number generator state
         */
        std::string const& getRandState() const
        {
            return m_randState;
        }

        /**
         * \brief A function that returns the number of parameters in the log-posterior
         * @return the number of parameters in the log-posterior
         */
        size_t numParams() const
        {
            return m_numParams;
        }

        /**
         * \brief A function that returns false when the required number of samples are taken
         * @return false when the required number of samples are taken, otherwise true
         */
        bool continueSampling() const
        {
            return m_continueSampling;
        }

        /**
         * \brief A function that returns the packet-size of the sampling process
         * @return returns the packet-size of the sampling process
         */
        size_t packetSize() const
        {
            return m_packetSize;
        }

    private:

        /**
         * \brief A function that renames the chain file upon taking the required number of samples
         */
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

        /**
         * \breif A function for serializing the data
         * \tparam Archive the archive type
         * \param ar the archive
         * \param version version number
         */
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

        size_t m_numParams; /**< number of parameters in the log-posterior */
        size_t m_packetSize; /**< packet-size of the sampling process */
        size_t m_numBurn; /**< number of samples to be burned */
        size_t m_numSamples; /**< number of samples to be taken after burning */
        std::string m_rootPathStr; /**< path to output files */
        bool m_consoleOutput; /**< a flag for controlling the console output */

        size_t m_samplesTaken; /**< samples taken so far */
        size_t m_samplesBurned; /**< samples burned so far */

        bool m_continueSampling; /**< continue sampling or not? */

        std::vector<realScalarType> m_startPoint; /**< start-point of the MCMC */
        std::string m_randState; /**< current random number state */
        realScalarType m_accRate; /**< current acceptance rate */

        std::string m_archiveOutFileName; /**< ouput file name of the archive */
        std::string m_chainOutFileName; /**< ouput file name of the MCMC chains */
    };

}//namespace control
}//namespace mpp

#endif //MPP_FINITESAMPLES_HPP

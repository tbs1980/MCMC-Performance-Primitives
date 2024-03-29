#ifndef MPP_WRITEALLPARAMS_HPP
#define MPP_WRITEALLPARAMS_HPP

namespace mpp{ namespace IO{
    /**
     * \ingroup Hamiltonian
     *
     * \class IOWriteAllParams
     *
     * \brief A class for managing the IO.
     *
     * \tparam _realScalarType real matrix type
     *
     * This class manages the IO of the sampler. In particular this class
     * writes the states of all the parameters to the chain output file.
     */
    template<typename _realScalarType>
    class IOWriteAllParams
    {
    public:

        /**
         * \typedef _realScalarType realScalarType;
         * \brief the floating point type
         */
        typedef _realScalarType realScalarType;

        /**
         * \typedef typename Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType;
         * \brief real vector type
         */
        typedef typename Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType;

        /**
         * \typedef typename Eigen::Matrix<realScalarType, Eigen::Dynamic, Eigen::Dynamic> realMatrixType;
         * \brief real matrix type
         */
        typedef typename Eigen::Matrix<realScalarType, Eigen::Dynamic, Eigen::Dynamic> realMatrixType;

        /**
         * \typedef typename realVectorType::Index indexType;
         * \brief integral type
         */
        typedef typename realVectorType::Index indexType;

        static_assert(std::is_floating_point<realScalarType>::value,
            "PARAMETER SHOULD BE A FLOATING POINT TYPE");

        /**
         * \brief The default constructor
         */
        IOWriteAllParams()
        :mThinLength(1),mPacketSize(0)
        {
        }

        /**
         * \brief A constructor that sets up the IO for the sampler
         * \param  fileName The name of the output file
         */
        explicit IOWriteAllParams(std::string const& fileName,indexType const thinLength = indexType(1),indexType const packetSize = indexType(0))
        :m_fileName(fileName),m_delimiter(std::string(",")),m_precision(10)
        ,mThinLength(thinLength),mPacketSize(packetSize)
        {
            BOOST_ASSERT_MSG(thinLength>0,"Thin length should be a positive integer.");
            BOOST_ASSERT_MSG(packetSize>=0,"Packet-size should be a positive integer or zero.");
        }

        /**
         * \brief A constructor that sets up the IO for the sampler
         * \param fileName The name of the output file
         * \param delimiter The delimiter between two numbers e.g. a coma
         * \param precision The precision with which the output is written
         */
        IOWriteAllParams(std::string const& fileName,std::string const & delimiter,
            unsigned int const precision,indexType const thinLength = indexType(1)
            ,indexType const packetSize = indexType(0))
        :m_fileName(fileName),m_delimiter(delimiter),m_precision(precision)
        ,mThinLength(thinLength),mPacketSize(packetSize)
        {
            BOOST_ASSERT_MSG(thinLength>0,"Thin length should be a positive integer.");
            BOOST_ASSERT_MSG(packetSize>=0,"Packet-size should be a positive integer or zero.");
        }

        /**
         * \brief The copy constructor
         * \param other The class from which the members are to be copied
         */
        IOWriteAllParams(IOWriteAllParams  const & other)
        {
            m_fileName = other.m_fileName;
            m_delimiter = other.m_delimiter;
            m_precision = other.m_precision;
            mThinLength = other.mThinLength;
            mPacketSize = other.mPacketSize;
        }

        /**
         * \breif The default destructor
         */
        ~IOWriteAllParams()
        {
            m_file.close();
        }

        /**
         * \brief write the samples to a file
         * \param samples the MCMC samples from the sampler
         */
        void write(realMatrixType const& samples,realVectorType const& logPostVals)
        {
            BOOST_ASSERT_MSG(samples.rows() == logPostVals.rows(),
                "Both samples and the logPostVals should have the same number of rows");

            if(!m_file.is_open())
            {
                m_file.open(m_fileName.c_str(),std::ios::app);
            }

            if(m_file.is_open())
            {
                m_file<<std::scientific;
                m_file<<std::setprecision(m_precision);
                //for(indexType i=0;i<samples.rows();++i)
                indexType i=0;
                while(i<samples.rows())
                {
                    //write the log posterior at the beginning
                    m_file<<logPostVals(i)<<m_delimiter;
                    //then write the chain values for every dimension
                    for(indexType j=0;j<samples.cols()-1;++j)
                    {
                        m_file<<samples(i,j)<<m_delimiter;
                    }
                    m_file<<samples(i,(samples.cols()-1) )<<std::endl;
                    i = i + mThinLength;
                    if(mPacketSize >0 and i>=mPacketSize)
                    {
                        break;
                    }
                }
            }
            else
            {
                std::string message = std::string("Error in opening the file ")+m_fileName;
                throw std::runtime_error(message);
            }
        }

        /**
         * \brief Return the output file name
         * \return  the output file name
         */
        inline std::string getFileName(void) const
        {
            return m_fileName;
        }

        /**
         * \brief Set the output file name
         * \param fileName the output file name
         */
        inline void setFileName(std::string const& fileName)
        {
            m_fileName = fileName;
        }

    private:
        std::string m_fileName; /**< the file output file name */
        std::ofstream m_file; /**< output file handler*/
        std::string m_delimiter; /**< delimiter */
        unsigned int m_precision; /**< precision */
        indexType mThinLength;
        indexType mPacketSize;

    };

}//namespace IO
}//namespace mpp

#endif //MPP_WRITEALLPARAMS_HPP

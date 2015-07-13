#ifndef MPP_CANONICALMCMC_HPP
#define MPP_CANONICALMCMC_HPP

namespace mpp{ namespace sampler{

    /**
     * \class canonicalMCMCSampler
     *
     * \tparam _engineType MCMC engine type
     * \tparam _ctrlrType controller type
     * \tparam _IOType input-output type
     *
     * \brief A class for performing MCMC sampling.
     *
     * This class performs sampling given the MCMC engine, controller and the IO. The
     * controller will stop MCMC when the stopping crieterion is satisfied. The IO will
     * be used to write the chains to the disk.
     */
    template<class _engineType,class _ctrlrType,class _IOType>
    class canonicalMCMCSampler
    {
    public:

        /**
         * \typedef _engineType engineType
         * \brief MCMC engine type
         */
        typedef _engineType engineType;

        /**
         * \typedef _ctrlrType ctrlrType
         * \brief controller type
         */
        typedef _ctrlrType ctrlrType;

        /**
         * \typedef _IOType IOType
         * \brief input-output type
         */
        typedef _IOType IOType;

        /**
         * \typedef typename engineType::realMatrixType realMatrixType
         * \brief defines the real floating point matrix type
         */
        typedef typename engineType::realMatrixType realMatrixType;

        /**
         * \typedef typename engineType::realVectorType realVectorType
         * \brief defines the real floating point vector type
         */
        typedef typename engineType::realVectorType realVectorType;

        /**
         * \typedef typename engineType::realScalarType realScalarType
         * \brief defines the real floating point type
         */
        typedef typename engineType::realScalarType realScalarType;

        static_assert(std::is_floating_point<realScalarType>::value,
            "PARAMETER SHOULD BE A FLOATING POINT TYPE");

        /**
         * \brief A function that performs the sampling
         * @param eng   MCMC engine
         * @param ctrlr controller
         * @param IO    input-output
         *
         * This function calls the generate() function from the MCMC engine repetedly
         * until the stopping crieterion is statisfied. The random number state and
         * acceptance rate of the MCMC chain is callled once each iteration using
         * getRandState() and getAcceptanceRate() functions. The dump() function
         * is used to pass information to the controller and finally write() function
         * from IO is used to write chains to the disk.
         */
        static void run(engineType & eng,ctrlrType & ctrlr,IOType & IO)
        {
            while( ctrlr.continueSampling() )
            {
                // this is kept inside the while loop so that a resume data mismatch
                // can be detected and the sampling stopped without an assertion error.
                BOOST_ASSERT_MSG(eng.numParams() == ctrlr.numParams(),
                    "Dimensionality mismatch");

                realMatrixType samples(ctrlr.packetSize(),ctrlr.numParams());
                realVectorType logPostVals(ctrlr.packetSize());

                eng.generate(samples,logPostVals);

                std::string randState = eng.getRandState();
                realScalarType accRate = eng.getAcceptanceRate();

                ctrlr.dump(samples,accRate,randState);

                IO.write(samples,logPostVals);
            }
        }
    };

}//Hamiltonian
}//namespace mpp

#endif //MPP_CANONICALMCMC_HPP

#ifndef MPP_CANONICALMCMC_HPP
#define MPP_CANONICALMCMC_HPP

namespace mpp{ namespace sampler{

    template<class _engineType,class _ctrlrType,class _IOType>
    class canonicalMCMCSampler
    {
    public:
        typedef _engineType engineType;
        typedef _ctrlrType ctrlrType;
        typedef _IOType IOType;
        typedef typename engineType::realMatrixType realMatrixType;
        typedef typename engineType::realVectorType realVectorType;
        typedef typename engineType::realScalarType realScalarType;

        static_assert(std::is_floating_point<realScalarType>::value,
            "PARAMETER SHOULD BE A FLOATING POINT TYPE");

        static void run(engineType & eng,ctrlrType & ctrlr,IOType & IO)
        {
            BOOST_ASSERT_MSG(eng.numParams() == ctrlr.numParams(),
                "Dimensionality mismatch");

            while( ctrlr.continueSampling() )
            {
                realMatrixType samples(ctrlr.packetSize(),ctrlr.numParams());
                realVectorType logPostVals(ctrlr.packetSize());

                eng.generate(samples,logPostVals);

                std::string randState = eng.getRandState();
                realScalarType accRate = eng.getAcceptanceRate();
                realVectorType startPoint = eng.getStartPoint();

                ctrlr.dump(samples,accRate,randState);

                IO.write(samples,logPostVals);
            }
        }
    };

}//Hamiltonian
}//namespace mpp

#endif //MPP_CANONICALMCMC_HPP

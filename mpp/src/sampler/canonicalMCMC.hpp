#ifndef MPP_CANONICALMCMC_HPP
#define MPP_CANONICALMCMC_HPP

namespace mpp{ namespace sampler{

    template<class _engineType,class _IOType,class _ctrlrType,class _tunerType>
    class canonicalMCMCSampler
    {
    public:
        typedef _engineType engineType;
        typedef _IOType IOType;
        typedef _ctrlrType ctrlrType;
        typedef typename engineType::realMatrixType realMatrixType;
        typedef typename engineType::realVectorType realVectorType;
        typedef typename engineType::realScalarType realScalarType;

        static_assert(std::is_floating_point<realScalarType>::value,
            "PARAMETER SHOULD BE A FLOATING POINT TYPE");

        static void run(engineType & eng,IOType & IO,runCtrlType & ctrlr,tunerType & tuner)
        {
            BOOST_ASSERT_MSG(eng.numParams() == ctrlr.numParams(),
                "Dimensionality mismatch");
            BOOST_ASSERT_MSG(eng.numParams() == tuner.numParams(),
                "Dimensionality mismatch");

            while( ctrlr.continueSampling() )
            {
                realMatrixType samples(ctrlr.packetSize(),ctrlr.numParams());
                realVectorType logPostVals(ctrlr.packetSize());

                eng.generate(samples,logPostVals);

                std::stringstream randState = eng.getRandState(randState);
                realScalarType accRate = eng.getAcceptanceRate();
                realVectorType startPoint = eng.getStartPoint();

                ctrlr.dump(startPoint,randState,accRate);

                IO.write(samples,logPostVals);

                tuner.tune(samples,accRate);
            }
        }
    };

}//Hamiltonian
}//namespace mpp

#endif //MPP_CANONICALMCMC_HPP

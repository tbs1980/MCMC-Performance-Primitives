#define BOOST_TEST_MODULE Control
#define BOOST_TEST_DYN_LINK
#define BOOST_ALL_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <mpp/control>

template<typename realScalarType>
void testFiniteSamples(void)
{
    typedef mpp::control::finiteSamplesControl<realScalarType> controlType;
    typedef typename controlType::realVectorType realVectorType;
    typedef typename controlType::realMatrixType realMatrixType;
    typedef mpp::utils::randomSTD<realScalarType> rVGenType;
    typedef typename rVGenType::seedType seedType;

    size_t const numParams = 10;
    size_t const packetSize = 100;
    size_t const numBurn = 100;
    size_t const numSamples = 100;
    std::string const rootPathStr("./testFiniteSamplesControl");
    bool const consoleOutput = true;
    realVectorType const startPoint = realVectorType::Zero(numParams);
    seedType seed = 0;
    rVGenType rvGen(seed);
    std::stringstream randStateStrm;
    rvGen.getState(randStateStrm);
    std::string const randState = randStateStrm.str();
    realMatrixType const samples = realMatrixType::Random(packetSize,numParams);
    realScalarType const accRate = 1.;


    controlType ctrl(numParams, packetSize, numBurn, numSamples, rootPathStr,
        consoleOutput,startPoint,randState);

    ctrl.dump(samples,accRate,randState);
    //ctrl.dump(samples,accRate,randState);
    //ctrl.dump(samples,accRate,randState);
}

BOOST_AUTO_TEST_CASE(finiteSamples)
{
    testFiniteSamples<float>();
}

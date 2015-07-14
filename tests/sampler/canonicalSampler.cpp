/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


#define BOOST_TEST_MODULE GaussLogPost
#define BOOST_TEST_DYN_LINK
#define BOOST_ALL_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <mpp/Hamiltonian>
#include <mpp/control>
#include <mpp/IO>
#include <mpp/sampler>

template<typename realScalarType>
void testCanonical(void)
{
    // define and engine
    typedef mpp::utils::GaussPotentialEnergy<realScalarType> potEngType;
    typedef mpp::Hamiltonian::GaussKineticEnergy<realScalarType> kinEngType;
    typedef mpp::utils::randomSTD<realScalarType> rVGenType;
    typedef mpp::Hamiltonian::leapfrog leapfrogIntegratorPolicy;
    typedef mpp::Hamiltonian::canonicalHMC<rVGenType,potEngType,kinEngType,
        leapfrogIntegratorPolicy> canonicalHMCType;
    typedef typename potEngType::realVectorType realVectorType;
    typedef typename potEngType::realMatrixType realMatrixType;
    typedef typename realMatrixType::Index indexType;
    typedef typename rVGenType::seedType seedType;
    // define control
    typedef mpp::control::finiteSamplesControl<realScalarType> controlType;
    // define IO
    typedef mpp::IO::IOWriteAllParams<realScalarType> IOType;
    // define the sampler
    typedef mpp::sampler::canonicalMCMCSampler<canonicalHMCType,controlType,IOType>
        samplerType;

    // define the diemensionaligy of the problem
    size_t const numParams = 10;

    // make a Gaussian posterior distribution
    realVectorType mu = realVectorType::Zero(numParams);
    realMatrixType sigmaInv = realMatrixType::Identity(numParams,numParams);
    potEngType G(mu,sigmaInv);

    // define a kinetic energy type
    realMatrixType mInv = realMatrixType::Identity(numParams,numParams);
    kinEngType K(mInv);

    // define the step size and the number of steps for the integrator
    realScalarType const maxEps = 1;
    indexType const maxNumsteps = 10;

    // define the start point
    realVectorType startPoint = realVectorType::Random(numParams);

    // define a random number seed
    seedType seed = 0;
    rVGenType rvGen(seed);

    // define the Hamiltonian Monte Carlo
    canonicalHMCType canonHMC(maxEps,maxNumsteps,startPoint,rvGen,G,K);

    // define the finite samples control
    size_t const packetSize = 100;
    size_t const numBurn = 0;
    size_t const numSamples = 1000;
    std::string const rootPathStr("./testCanonicalSampler");
    bool const consoleOutput = true;

    std::string randState = canonHMC.getRandState();

    controlType ctrl(numParams, packetSize, numBurn, numSamples, rootPathStr,
        consoleOutput,startPoint,randState);

    // if resuming from pervious state, set the start point here
    startPoint = ctrl.getStartPoint();
    randState = ctrl.getRandState();
    canonHMC.setStartPoint(startPoint);
    canonHMC.setRandState(randState);

    // define IO
    const std::string outFileName = ctrl.getChainFileName();
    const std::string delimiter(",");
    const unsigned int precision = 10;

    IOType writeAllIO(outFileName,delimiter,precision);

    // define the sampler
    samplerType::run(canonHMC,ctrl,writeAllIO);

}

template<typename realScalarType>
void testCanonicalDiag(void)
{
    // define and engine
    typedef mpp::utils::GaussPotentialEnergyDiag<realScalarType> potEngType;
    typedef mpp::Hamiltonian::GaussKineticEnergyDiag<realScalarType> kinEngType;
    typedef mpp::utils::randomSTD<realScalarType> rVGenType;
    typedef mpp::Hamiltonian::leapfrog leapfrogIntegratorPolicy;
    typedef mpp::Hamiltonian::canonicalHMC<rVGenType,potEngType,kinEngType,
        leapfrogIntegratorPolicy> canonicalHMCType;
    typedef typename potEngType::realVectorType realVectorType;
    typedef typename potEngType::realDiagMatrixType realDiagMatrixType;
    typedef typename realDiagMatrixType::Index indexType;
    typedef typename rVGenType::seedType seedType;
    // define control
    typedef mpp::control::finiteSamplesControl<realScalarType> controlType;
    // define IO
    typedef mpp::IO::IOWriteAllParams<realScalarType> IOType;
    // define the sampler
    typedef mpp::sampler::canonicalMCMCSampler<canonicalHMCType,controlType,IOType>
        samplerType;

    // define the diemensionaligy of the problem
    size_t const numParams = 10;

    // make a Gaussian posterior distribution
    realVectorType mu = realVectorType::Zero(numParams);
    realDiagMatrixType sigmaInv = realVectorType::Ones(numParams);
    potEngType G(mu,sigmaInv);

    // define a kinetic energy type
    realDiagMatrixType mInv = realVectorType::Ones(numParams);
    kinEngType K(mInv);

    // define the step size and the number of steps for the integrator
    realScalarType const maxEps = 1;
    indexType const maxNumsteps = 10;

    // define the start point
    realVectorType startPoint = realVectorType::Random(numParams);

    // define a random number seed
    seedType seed = 0;
    rVGenType rvGen(seed);

    // define the Hamiltonian Monte Carlo
    canonicalHMCType canonHMC(maxEps,maxNumsteps,startPoint,rvGen,G,K);

    // define the finite samples control
    size_t const packetSize = 100;
    size_t const numBurn = 0;
    size_t const numSamples = 1000;
    std::string const rootPathStr("./testCanonicalSampler");
    bool const consoleOutput = true;

    std::string randState = canonHMC.getRandState();

    controlType ctrl(numParams, packetSize, numBurn, numSamples, rootPathStr,
        consoleOutput,startPoint,randState);

    // if resuming from pervious state, set the start point here
    startPoint = ctrl.getStartPoint();
    randState = ctrl.getRandState();
    canonHMC.setStartPoint(startPoint);
    canonHMC.setRandState(randState);

    // define IO
    const std::string outFileName = ctrl.getChainFileName();
    const std::string delimiter(",");
    const unsigned int precision = 10;

    IOType writeAllIO(outFileName,delimiter,precision);

    // define the sampler
    samplerType::run(canonHMC,ctrl,writeAllIO);

}


BOOST_AUTO_TEST_CASE(canonical)
{
    testCanonical<float>();
    testCanonical<double>();
    testCanonical<long double>();

    testCanonicalDiag<float>();
    testCanonicalDiag<double>();
    testCanonicalDiag<long double>();
}

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define BOOST_TEST_MODULE GaussLogPost
#define BOOST_TEST_DYN_LINK
#define BOOST_ALL_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <mpp/Hamiltonian>
#include <mpp/tempering>
#include <mpp/control>
#include <mpp/IO>
#include <mpp/sampler>
#include <mpp/UI>

template<typename realScalarType>
void testParallelTemperedHamiltonianSampler()
{
    // define a log posterior
    typedef mpp::utils::GaussPotentialEnergyDiag<realScalarType> logPosteriorType;

    // define the interface for the sampler
    //typedef mpp::canonicalHamiltonianSampler<logPosteriorType> samplerType;
    typedef mpp::parallelTemperingHamiltonianSampler<logPosteriorType> samplerType;

    typedef typename samplerType::realVectorType realVectorType;

    typedef typename logPosteriorType::realDiagMatrixType realDiagMatrixType;

    typedef typename samplerType::seedType seedType;

    // define the diemensionaligy of the problem
    size_t const numParams = 10;

    // make a Gaussian posterior distribution
    realVectorType mu = realVectorType::Zero(numParams);
    realDiagMatrixType sigmaInv = realDiagMatrixType::Ones(numParams);
    logPosteriorType G(mu,sigmaInv);

    // define the step size and the number of steps for the integrator
    realScalarType const maxEps = 1;
    size_t const maxNumSteps = 10;

    // define the start point
    realVectorType startPoint = realVectorType::Random(numParams);

    // define a random number seed
    seedType randSeed = 0;

    // define the finite samples control
    size_t const packetSize = 100;
    size_t const numBurn = 0;
    size_t const numSamples = 1000;
    std::string const rootPathStr("./testParallelTemperedHamiltonianSampler");
    bool const consoleOutput = true;

    // define IO
    const std::string delimiter(",");
    const unsigned int precision = 10;

    // diagonal elements of the mass matrix
    realDiagMatrixType MInv = realDiagMatrixType::Ones(numParams);

    // define parameters for the parallel tempering
    realScalarType const powBaseFact = 0.005;
    realScalarType const swapRatio = 0.5;
    size_t const numChains = 10;

    samplerType prlTempHamiltSampler(
        G,
        numParams,
        maxEps,
        maxNumSteps,
        startPoint,
        randSeed,
        packetSize,
        numBurn,
        numSamples,
        rootPathStr,
        consoleOutput,
        delimiter,
        precision,
        MInv,
        powBaseFact,
        swapRatio,
        numChains
    );

    // finally run the sampler
    prlTempHamiltSampler.run();

}

BOOST_AUTO_TEST_CASE(parallelTemperedHamiltonianSampler)
{
    BOOST_TEST_MESSAGE( "Testing parallel tempered Hamiltonian sampler for float,dobule and long double." );

    testParallelTemperedHamiltonianSampler<float>();
    testParallelTemperedHamiltonianSampler<double>();
    testParallelTemperedHamiltonianSampler<long double>();
}

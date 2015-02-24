/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define BOOST_TEST_MODULE CanonicalHMC
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <mpp/Hamiltonian>

template<typename realScalarType>
void testCanonicalHMCDim10(void)
{
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

    const indexType N = 10;

    realVectorType mu = realVectorType::Zero(N);
    realMatrixType sigmaInv = realMatrixType::Identity(N,N);
    realMatrixType mInv = realMatrixType::Identity(N,N);
    realVectorType q0 = realVectorType::Random(N);

    const realScalarType maxEps = 1;
    const indexType maxNumsteps = 10;

    potEngType G(mu,sigmaInv);
    kinEngType K(mInv);

    seedType seed = 0;

    const indexType numsamples = 1000;
    realMatrixType samples(numsamples,N);
    realVectorType logPostVals = realVectorType::Zero(numsamples);

    canonicalHMCType canonHMC(maxEps,maxNumsteps,q0,seed,G,K);
    canonHMC.generate(samples,logPostVals);

    BOOST_REQUIRE(0.91 < canonHMC.getAcceptanceRate() and canonHMC.getAcceptanceRate() < 0.95);
}

BOOST_AUTO_TEST_CASE(CanonicalHMCDim10)
{
    testCanonicalHMCDim10<float>();
}

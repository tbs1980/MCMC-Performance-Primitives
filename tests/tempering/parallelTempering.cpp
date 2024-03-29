/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define BOOST_TEST_MODULE CanonicalHMC
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <mpp/Hamiltonian>
#include <mpp/tempering>

//http://en.wikipedia.org/wiki/Rosenbrock_function

//define the Rosenbrock density in 2D
template<typename _realScalarType>
class Rosenbrock
{
public:
    typedef _realScalarType realScalarType;
    typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType;
    typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realDiagMatrixType;
    typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, Eigen::Dynamic> realMatrixType;
    typedef typename realVectorType::Index indexType;

    Rosenbrock(realScalarType a=1,realScalarType b=100)
    :mA(a),mB(b)
    {

    }

    void value(realVectorType const & q, realScalarType & val) const
    {
        BOOST_ASSERT(q.rows()==2);

        val = -( mA-q(0) )*( mA-q(0) ) - mB*( q(1)-q(0)*q(0) )*( q(1)-q(0)*q(0) );
    }

    void derivs(realVectorType const & q,realVectorType & dq) const
    {
        BOOST_ASSERT(q.rows()==2);
        BOOST_ASSERT(dq.rows()==2);

        dq(0) = 2*( mA-q(0) ) + 4*mB*q(0)*( q(1)-q(0)*q(0) );
        dq(1) = -2*mB*( q(1) - q(0)*q(0) );

    }

    inline indexType numDims() const
    {
        return indexType(2);
    }

private:
    realScalarType mA;
    realScalarType mB;
};

template<typename realScalarType>
void testParallelTemperingHMCDim10(void)
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
    typedef mpp::prltemp::powerLawTemperature<realScalarType> chainTempType;
    typedef mpp::prltemp::parallelTemperingMCMC<canonicalHMCType,chainTempType> parallelTemperingMCMCType;

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
    rVGenType rvGen(seed);

    canonicalHMCType canonHMC(maxEps,maxNumsteps,q0,rvGen,G,K);

    size_t const numChains = 8;

    realScalarType swapRatio = 0.5;
    chainTempType chainTemps(0.005,numChains);

    parallelTemperingMCMCType paraTempMCMC(canonHMC,swapRatio,chainTemps);

    paraTempMCMC.generate(samples,logPostVals);


    BOOST_REQUIRE(0.95 < paraTempMCMC.getAcceptanceRate() and paraTempMCMC.getAcceptanceRate() < 0.98);
}

template<typename realScalarType>
void testParallelTempering2DRosenbrock(void)
{
    typedef Rosenbrock<realScalarType> potEngType;
    typedef mpp::Hamiltonian::GaussKineticEnergy<realScalarType> kinEngType;
    typedef mpp::utils::randomSTD<realScalarType> rVGenType;
    typedef mpp::Hamiltonian::leapfrog leapfrogIntegratorPolicy;
    typedef mpp::Hamiltonian::canonicalHMC<rVGenType,potEngType,kinEngType,
        leapfrogIntegratorPolicy> canonicalHMCType;
    typedef typename potEngType::realVectorType realVectorType;
    typedef typename potEngType::realMatrixType realMatrixType;
    typedef typename realMatrixType::Index indexType;
    typedef typename rVGenType::seedType seedType;
    typedef mpp::prltemp::powerLawTemperature<realScalarType> chainTempType;
    typedef mpp::prltemp::parallelTemperingMCMC<canonicalHMCType,chainTempType> parallelTemperingMCMCType;

    const indexType N = 2;

    realScalarType a = 1;
    realScalarType b =100;

    realVectorType mu = realVectorType::Zero(N);
    realMatrixType sigmaInv = realMatrixType::Identity(N,N);
    realMatrixType mInv = realMatrixType::Identity(N,N)*2.5e-3;
    realVectorType q0 = realVectorType::Random(N);


    const realScalarType maxEps = 1;
    const indexType maxNumsteps = 10;

    potEngType G(a,b);
    kinEngType K(mInv);

    seedType seed = 0;

    indexType const numChains = 8;

    const indexType numsamples = 1000;
    realMatrixType samples(numsamples*numChains,N);
    realVectorType logPostVals = realVectorType::Zero(numsamples*numChains);
    rVGenType rvGen(seed);

    canonicalHMCType canonHMC(maxEps,maxNumsteps,q0,rvGen,G,K);


    realScalarType swapRatio = 0.5;
    chainTempType chainTemps(0.0001,(size_t)numChains);

    parallelTemperingMCMCType paraTempMCMC(canonHMC,swapRatio,chainTemps);

    paraTempMCMC.generate(samples,logPostVals);


    BOOST_REQUIRE(0.96 < paraTempMCMC.getAcceptanceRate() and paraTempMCMC.getAcceptanceRate() < 1);

}


BOOST_AUTO_TEST_CASE(parallelTempering2D4BlobGauss)
{
    testParallelTemperingHMCDim10<float>();
    testParallelTemperingHMCDim10<double>();
    testParallelTemperingHMCDim10<long double>();
}


BOOST_AUTO_TEST_CASE(parallelTempering2DRosenbrock)
{
    testParallelTempering2DRosenbrock<float>();
    testParallelTempering2DRosenbrock<double>();
    testParallelTempering2DRosenbrock<long double>();
}

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


#define BOOST_TEST_MODULE GaussLogPost
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <mpp/utils>

template<typename realScalarType>
void testGaussianPotentialEnergy(void)
{
    typedef typename mpp::utils::GaussPotentialEnergy<realScalarType> potEngType;
    typedef typename potEngType::realVectorType realVectorType;
    typedef typename potEngType::realMatrixType realMatrixType;
    typedef typename realMatrixType::Index indexType;

    const indexType N = 100;

    realVectorType mu = realVectorType::Zero(N);
    realMatrixType sigmaInv = realMatrixType::Identity(N,N);
    realVectorType q = realVectorType::Random(N);

    potEngType G(mu,sigmaInv);

    realVectorType dq = sigmaInv*(mu-q);
    realScalarType val = -0.5*(mu-q).transpose()*sigmaInv*(mu-q);

    realVectorType dqTest=realVectorType::Zero(N);
    realScalarType valTest = 0;

    G.value(q,valTest);
    BOOST_CHECK_EQUAL(val,valTest);

    G.derivs(q,dqTest);
    for(indexType i=0;i<N;++i)
    {
        BOOST_CHECK_EQUAL(dq(i),dqTest(i));
    }
}

template<typename realScalarType>
void testGaussianPotentialEnergyDiag(void)
{
    typedef typename mpp::utils::GaussPotentialEnergyDiag<realScalarType> potEngType;
    typedef typename potEngType::realVectorType realVectorType;
    typedef typename potEngType::realDiagMatrixType realDiagMatrixType;
    typedef typename realDiagMatrixType::Index indexType;

    const indexType N = 100;

    realVectorType mu = realVectorType::Zero(N);
    realVectorType q = realVectorType::Random(N);
    realDiagMatrixType sigmaInv(N);
    for(indexType i=0;i<N;++i)
    {
        sigmaInv(i)=1;
    }

    potEngType G(mu,sigmaInv);

    realVectorType dq = sigmaInv.cwiseProduct(mu-q);
    realScalarType val = -0.5*(mu-q).transpose()*(sigmaInv.cwiseProduct(mu-q));

    realScalarType valTest = 0;
    realVectorType dqTest = realVectorType::Zero(N);

    G.value(q,valTest);
    BOOST_CHECK_EQUAL(val,valTest);

    G.derivs(q,dqTest);
    for(indexType i=0;i<N;++i)
    {
        BOOST_CHECK_EQUAL(dq(i),dqTest(i));
    }
}

BOOST_AUTO_TEST_CASE(GaussianPotentialEnergy)
{
    testGaussianPotentialEnergy<float>();
    testGaussianPotentialEnergy<double>();
    testGaussianPotentialEnergy<long double>();
}

BOOST_AUTO_TEST_CASE(GaussianPotentialEnergyDiag)
{
    testGaussianPotentialEnergyDiag<float>();
    testGaussianPotentialEnergyDiag<double>();
    testGaussianPotentialEnergyDiag<long double>();
}

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define BOOST_TEST_MODULE kineticEnergy
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <mpp/Hamiltonian>

template<typename realScalarType>
void testNDGeneralGauss(void)
{
    typedef typename mpp::Hamiltonian::GaussKineticEnergy<realScalarType> kinEngType;
    typedef typename kinEngType::realVectorType realVectorType;
    typedef typename kinEngType::realMatrixType realMatrixType;
    typedef typename realMatrixType::Index indexType;

    const indexType N = 100;

    realVectorType x = realVectorType::Random(N);
    realVectorType g = realVectorType::Zero(N);
    realMatrixType mInv = realMatrixType::Identity(N,N);
    realScalarType val = 0;
    kinEngType phi(mInv);

    realScalarType valTest=-0.5*x.transpose()*mInv*x;

    phi.value(x,val);
    BOOST_CHECK_EQUAL(val,valTest);

    phi.derivs(x,g);
    for(indexType i=0;i<N;++i)
    {
        BOOST_CHECK_EQUAL(g(i),-x(i));
    }

    BOOST_CHECK_EQUAL(N,phi.numDims());
}

template<typename realScalarType>
void testNDDiagGauss(void)
{
    typedef typename mpp::Hamiltonian::GaussKineticEnergyDiag<realScalarType> kinEngType;
    typedef typename kinEngType::realVectorType realVectorType;
    typedef typename kinEngType::realDiagMatrixType realDiagMatrixType;
    typedef typename realDiagMatrixType::Index indexType;

    const indexType N = 1000000;

    realVectorType x = realVectorType::Random(N);
    realVectorType g = realVectorType::Zero(N);
    realDiagMatrixType mInv(N);
    for(indexType i=0;i<N;++i)
    {
        mInv(i)=1;
    }
    realScalarType val = 0;
    kinEngType phi(mInv);

    realScalarType valTest = -0.5*x.transpose()*(mInv.cwiseProduct(x));

    phi.value(x,val);
    BOOST_CHECK_EQUAL(val,valTest);

    phi.derivs(x,g);
    for(indexType i=0;i<N;++i)
    {
        BOOST_CHECK_EQUAL(g(i),-x(i));
    }

    BOOST_CHECK_EQUAL(N,phi.numDims());
}

template<typename realScalarType>
void testNDDiagGaussZeroRotate(void)
{
    typedef typename mpp::Hamiltonian::GaussKineticEnergyDiag<realScalarType> kinEngType;
    typedef typename kinEngType::realVectorType realVectorType;
    typedef typename kinEngType::realDiagMatrixType realDiagMatrixType;
    typedef typename realDiagMatrixType::Index indexType;

    const indexType N = 5;

    realVectorType x = realVectorType::Random(N);
    realVectorType g = realVectorType::Zero(N);
    realDiagMatrixType mInv(N);
    for(indexType i=0;i<N;++i)
    {
        mInv(i)=1;
    }

    //make the first value zero
    mInv(0)=0.;

    realScalarType val = 0;
    kinEngType phi(mInv);

    realScalarType valTest = -0.5*x.transpose()*(mInv.cwiseProduct(x));

    phi.value(x,val);
    BOOST_CHECK_EQUAL(val,valTest);

    phi.derivs(x,g);
    // check if we get zero for the frist one
    BOOST_CHECK_EQUAL(g(0),0);
    // rest of them are like berfore
    for(indexType i=1;i<N;++i)
    {
        BOOST_CHECK_EQUAL(g(i),-x(i));
    }

    BOOST_CHECK_EQUAL(N,phi.numDims());

    // check the rotation
    realVectorType p = realVectorType::Random(N);
    realVectorType pRot(p);
    phi.rotate(pRot);
    // sinnce the first mInv is zero, we expect to get a zero
    // although mathematically we should get infinity!
    BOOST_CHECK_EQUAL(pRot(0),0);
    // rest of them are like berfore
    for(indexType i=1;i<N;++i)
    {
        BOOST_CHECK_EQUAL(pRot(i),p(i));
    }

}

BOOST_AUTO_TEST_CASE(NDGeneralGauss)
{
    testNDGeneralGauss<float>();
    testNDGeneralGauss<double>();
    testNDGeneralGauss<long double>();
}

BOOST_AUTO_TEST_CASE(NDDiagGauss)
{
    testNDDiagGauss<float>();
    testNDDiagGauss<double>();
    testNDDiagGauss<long double>();
}

BOOST_AUTO_TEST_CASE(NDDiagGaussZeroRotate)
{
    testNDDiagGaussZeroRotate<float>();
    testNDDiagGaussZeroRotate<double>();
    testNDDiagGaussZeroRotate<long double>();
}

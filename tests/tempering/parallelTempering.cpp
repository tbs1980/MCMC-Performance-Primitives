/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define BOOST_TEST_MODULE CanonicalHMC
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <mpp/Hamiltonian>
#include <mpp/tempering>

template<typename _realScalarType>
class Gauss2D4Blobs
{
public:
    // http://www.lindonslog.com/example_code/tempering.cpp

    typedef _realScalarType realScalarType; /* THIS DEFINITION IS REQUIRED */
    typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> realDiagMatrixType;

    Gauss2D4Blobs()
    {
        mu1 = realVectorType::Ones(2)*-2.;
        mu2 = realVectorType::Ones(2)*2.;
        sigmaInv1 = realVectorType::Ones(2)*30.;
        sigmaInv2 = realVectorType::Ones(2)*30.;
    }

    void value(realVectorType const & q, double & val)
    {
        BOOST_ASSERT_MSG(q.rows()==2,"This posterior distribution is 2D only");

        /*
        realScalarType t1 = q(0);
        realScalarType t2 = q(1);

        realScalarType sigma2=0.001;
        realScalarType lik = std::exp(-(1/(2*sigma2))*((t1-1)*(t1-1) + (t2-1)*(t2-1) ))
            + std::exp(-(1/(2*sigma2))*((t1-0)*(t1-0) + (t2-0)*(t2-0) ))
            + std::exp(-(1/(2*sigma2))*((t1-1)*(t1-1) + (t2+1)*(t2+1) ))
            + std::exp(-(1/(2*sigma2))*((t1+1)*(t1+1) + (t2-1)*(t2-1) ))
            + std::exp(-(1/(2*sigma2))*((t1+1)*(t1+1) + (t2+1)*(t2+1) ));

        //sigma2=1000;
        //realScalarType prior =  std::exp((-1/(2*sigma2))*(t1*t1+t2*t2));

        val = std::log(lik) ;//+ std::log(prior);
        */


        val = std::exp(-0.5*(mu1-q).transpose()*sigmaInv1.cwiseProduct(mu1-q))
            + std::exp(-0.5*(mu2-q).transpose()*sigmaInv2.cwiseProduct(mu2-q));
        val = std::log(val);
    }

    void derivs(realVectorType const & q,realVectorType & dq) const
    {
        realScalarType p = std::exp(-0.5*(mu1-q).transpose()*sigmaInv1.cwiseProduct(mu1-q))
            + std::exp(-0.5*(mu2-q).transpose()*sigmaInv2.cwiseProduct(mu2-q));
    }

private:
    realVectorType mu1;
    realVectorType mu2;
    realDiagMatrixType sigmaInv1;
    realDiagMatrixType sigmaInv2;

};

template<typename realScalarType>
void makePlotData(void)
{
    typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType;
    typedef typename realVectorType::Index indexType;

    Gauss2D4Blobs<realScalarType> logPost;

    realScalarType t1Min = -5;
    realScalarType t1Max = 5;
    realScalarType t2Min = -5;
    realScalarType t2Max = 5;

    indexType numSteps = 200;

    realScalarType dt1 = (t1Max - t1Min)/(realScalarType) numSteps;
    realScalarType dt2 = (t2Max - t2Min)/(realScalarType) numSteps;

    std::ofstream fileOut;
    fileOut.open("./plot2d4Blob.dat",std::ios::trunc);

    for(indexType i=0;i<numSteps;++i)
    {
        realScalarType t1i = t1Min + i*dt1;
        for(indexType j=0;j<numSteps;++j)
        {
            realScalarType t2i = t2Min + j*dt2;
            realVectorType q(2);
            q(0) = t1i;
            q(1) = t2i;
            realScalarType val=0;
            logPost.value(q,val);
            fileOut<<t1i<<"\t"<<t2i<<"\t"<<val<<std::endl;
        }
        fileOut<<std::endl;
    }

    fileOut.close();
}

template<typename realScalarType>
void testTestParallelTempering2D4BlobGauss(void)
{

}

BOOST_AUTO_TEST_CASE(parallelTempering2D4BlobGauss)
{
    makePlotData<double>();
    testTestParallelTempering2D4BlobGauss<double>();
}

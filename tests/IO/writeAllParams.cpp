/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define BOOST_TEST_MODULE WriteAllParams
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <mpp/IO>

template<typename realScalarType>
void testMat10D(void)
{
    typedef mpp::IO::IOWriteAllParams<realScalarType> IOType;
    typedef typename IOType::realMatrixType realMatrixType;
    typedef typename IOType::realVectorType realVectorType;
    typedef typename realVectorType::Index indexType;

    const std::string outFileName("testIOWriteAllParams.dat");
    const std::string delimiter(",");
    const unsigned int precision = 10;
    const indexType thinLength = 2;

    IOType testIO(outFileName,delimiter,precision,thinLength);

    const indexType numSamples = 100;
    const indexType numParams = 1;

    realMatrixType samples = realMatrixType::Random(numSamples,numParams);
    realVectorType logPostVals = realVectorType::Random(numSamples);

    testIO.write(samples,logPostVals);
}

BOOST_AUTO_TEST_CASE(mat10D)
{
    testMat10D<float>();
}

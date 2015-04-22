/**
 * \file canonicalHMCSamplerGaussian.cpp
 *
 * This file explains how to compile an example code using mpp headers
 *
 * You will need
 * (1) Eigen http://eigen.tuxfamily.org
 * (2) Boost http://www.boost.org
 *
 * Eigen is a header only library and we only need path to Eigen/Core.
 *
 * From Boost we need
 * (1) serialization (e.g., libboost_serialization.so)
 * (2) filesystem (e.g., libboost_filesystem.so)
 * (3) system (e.g., libboost_system.so)
 * (4) log (e.g., libboost_log.so)
 * (5) thread (e.g., libboost_thread.so)
 * (6) date_time (e.g., libboost_date_time.so).
 *
 * These boost libraries and the corresponding headers are necessary for
 * linking this example correctly.
 *
 * We also need libpthread in order for the threads to work
 *
 * You can compile this example by
 *
 * g++  -pedantic -Wall -Wextra -Wfatal-errors -std=c++11 -g0 -O3 -Ipath_to_eigen_headers -Ipath_to_boost_headers -I../ canonicalHMCSamplerGaussian.cpp -o example_canonicalHMCSamplerGaussian -Lpath_to_boost_lib -Lpath_to_pthread -lboost_serialization -lboost_filesystem -lboost_system -lboost_log -lboost_thread -lboost_date_time -lpthread
 *
 * For example in my desktop, I will compile this by
 *
 * g++  -pedantic -Wall -Wextra -Wfatal-errors -std=c++11 -g0 -O3 -I/usr/include/eigen3 -I/usr/include -I../ canonicalHMCSamplerGaussian.cpp -o example_canonicalHMCSamplerGaussian -L/usr/lib/x86_64-linux-gnu -lboost_serialization -lboost_filesystem -lboost_system -lboost_log -lboost_thread -lboost_date_time -lpthread
 *
 * This will create an executable example_canonicalHMCSamplerGaussian
 */

#define BOOST_ALL_DYN_LINK // this is required for proper linking to boost
#include <mpp/core>


/**
 * \class GaussianLogPost
 *
 * \brief Multi-variate Gaussian distribution with no cross-correlation
 *
 * This class defines a Multi-variate Gaussian distribution with no cross-correlation
 * between parameters. In other words the covariance matrix is diagonal.
 */
class GaussianLogPost
{
public:

    /* We use Eigen for linear algegra. For example we define real-vector and
    a diagonal real-matrix below using Eigen facilities. If using other
    types for passing vectors, you will need to write an adaptor that converts
    the Eigen real-vector to your vector-type.
     */

    typedef double realScalarType; /* THIS DEFINITION IS REQUIRED */
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> realVectorType;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> realDiagMatrixType;
    typedef realVectorType::Index indexType;

    /**
     * \brief The default constructor that allocates no memory
     */
    GaussianLogPost()
    :mMu(0,0),mSigmaInv(0,0)
    {

    }

    /**
     * \brief A constructor that sets up the class
     *
     * \param mu mean of the distribution
     * \param sigmaInv inverse of the digonal covariance matrix, i.e., 1/variance(i)
     */
    GaussianLogPost(realVectorType const& mu,realDiagMatrixType const& sigmaInv)
    :mMu(mu),mSigmaInv(sigmaInv)
    {

    }

    /**
     * \brief compute the log-posterior probability
     *
     * \param q the position at which the log-post is required
     * \param val value of the log-posterior probability
     */
    inline void value(realVectorType const & q, double & val) const
    {
        val = -0.5*(mMu-q).transpose()*mSigmaInv.cwiseProduct(mMu-q);
    }

    /**
     * \brief compute the derivties of the log-post wrt q
     *
     * \param q the position at which the log-post derivs are sought
     * \param dq the derivatives to be returned
     */
    inline void derivs(realVectorType const & q,realVectorType & dq) const
    {
        dq = mSigmaInv.cwiseProduct(mMu-q);
    }

    /**
     * \brief return the number of dimensions of the posterior distribution
     *
     * \return the number of dimensions of the log-posterior distribution
     */
    inline indexType numDims() const
    {
        return mSigmaInv.rows();
    }

private:
    realVectorType mMu; /**< the mean of the distribution */
    realDiagMatrixType mSigmaInv; /**< the inverse of the covariance matrix */
};


int main(void)
{
    /*
    We will no define several components to the sampler
    (1) An EngineType. In this case Hamiltonian Monte Carlo Engine
    (2) A ControlType. e.g. A class that tells the sampler when to stop
    (3) An IOType. e.g. A class that write the output of to chains file

    We will use built-in classes for (1), (2) and (3)
     */

    // define the potential energy, i.e., the log-posterior
    typedef GaussianLogPost potEngType;
    // real vector for defining log-posterior, derive from the potEngType
    typedef typename potEngType::realVectorType realVectorType;
    // real diagonal-matrix for defining log-posterior , derive from the potEngType
    typedef typename potEngType::realDiagMatrixType realDiagMatrixType;

    // define a kinetic energy. we use the built-in Gaussian-Diagonal-KE
    typedef mpp::Hamiltonian::GaussKineticEnergyDiag<double> kinEngType;

    // define a random number generator type, we use the built-in RNG
    typedef mpp::utils::randomSTD<double> rVGenType;
    // define a seed type for RNG
    typedef typename rVGenType::seedType seedType;

    // define an descretisation policy for HMC
    typedef mpp::Hamiltonian::leapfrog leapfrogIntegratorPolicy;

    // define the HMC sampler, in this case canonicalHMC
    typedef mpp::Hamiltonian::canonicalHMC<rVGenType,potEngType,kinEngType,
        leapfrogIntegratorPolicy> canonicalHMCType;

    // define control type for controlling the sampler
    typedef mpp::control::finiteSamplesControl<double> controlType;

    // define IO type for defining the input-output stuff
    typedef mpp::IO::IOWriteAllParams<double> IOType;

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
    double const maxEps = 1;
    size_t const maxNumsteps = 10;

    // define the start point
    realVectorType startPoint = realVectorType::Random(numParams);

    // define a random number seed
    seedType seed = 0;

    // define the Hamiltonian Monte Carlo
    canonicalHMCType canonHMC(maxEps,maxNumsteps,startPoint,seed,G,K);

    // define the finite samples control
    // we burn numBurn samples and stop after numSamples are sampled
    size_t const packetSize = 100;
    size_t const numBurn = 0;
    size_t const numSamples = 1000;

    // define the output path
    std::string const rootPathStr("./exampleCanonicalSampler");

    // do we require output to console?
    bool const consoleOutput = true;

    // we ask HMC the current random number state
    // this is need for the controller
    std::string randState = canonHMC.getRandState();

    // define the controller
    controlType ctrl(numParams, packetSize, numBurn, numSamples, rootPathStr,
        consoleOutput,startPoint,randState);

    // get the name of the chain file from the controller
    const std::string outFileName = ctrl.getChainFileName();
    // what delimiter?
    const std::string delimiter(",");
    // what precision?
    const unsigned int precision = 10;
    // define the IO
    IOType writeAllIO(outFileName,delimiter,precision);

    // finally define the sampler
    samplerType::run(canonHMC,ctrl,writeAllIO);

    return 0;
}

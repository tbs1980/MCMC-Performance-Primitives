#define BOOST_ALL_DYN_LINK
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
    // define the potential energy, i.e., the log-posterior
    typedef GaussianLogPost potEngType;
    // real vector for defining log-posterior, derive from the potEngType
    typedef typename potEngType::realVectorType realVectorType;
    // real diagonal-matrix for defining log-posterior , derive from the potEngType
    typedef typename potEngType::realDiagMatrixType realDiagMatrixType;
    // define a kinetic energy. we use the built-in Gaussian-Diagonal-KE
    typedef mpp::Hamiltonian::GaussKineticEnergyDiag<double> kinEngType;
    // define the sampler using the User Interface
    typedef mpp::canonicalHamiltonianSampler<potEngType> samplerUIType;
    // define the seed type
    typedef samplerUIType::seedType seedType;

    // define the diemensionaligy of the problem
    size_t const numParams = 10;

    // make a Gaussian posterior distribution
    realVectorType mu = realVectorType::Zero(numParams);
    realDiagMatrixType sigmaInv = realVectorType::Ones(numParams);
    potEngType G(mu,sigmaInv);

    // define the step size and the number of steps for the integrator
    double const maxEps = 1;
    size_t const maxNumsteps = 10;

    // define the start point
    realVectorType startPoint = realVectorType::Random(numParams);

    // define a random number seed
    seedType seed = 0;

    // define the finite samples control
    // we burn numBurn samples and stop after numSamples are sampled
    size_t const packetSize = 100;
    size_t const numBurn = 0;
    size_t const numSamples = 1000;

    // define the output path
    std::string const rootPathStr("./exampleCanonicalSampler");

    // do we require output to console?
    bool const consoleOutput = true;

    // what delimiter?
    const std::string delimiter(",");
    // what precision?
    const unsigned int precision = 10;
    // define the IO

    // define a kinetic energy type
    realDiagMatrixType mInv = realVectorType::Ones(numParams);

    samplerUIType samplerUI(G,numParams,maxEps,maxNumsteps,startPoint,
        seed,packetSize,numBurn,numSamples,rootPathStr,consoleOutput,
        delimiter,precision,mInv);

    samplerUI.run();

    return 0;
}

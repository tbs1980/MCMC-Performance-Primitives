#define BOOST_ALL_DYN_LINK
#include <mpp/core>

#include <unsupported/Eigen/NumericalDiff>

// Generic functor
template<typename _Scalar, int NX=Eigen::Dynamic, int NY=Eigen::Dynamic>
struct Functor
{
  typedef _Scalar Scalar;
  enum {
    InputsAtCompileTime = NX,
    ValuesAtCompileTime = NY
  };
  typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
  typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
  typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

  int m_inputs, m_values;

  Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

};


/**
 * \class GaussianLogPost
 *
 * \brief Multi-variate Gaussian distribution with no cross-correlation
 *
 * This class defines a Multi-variate Gaussian distribution with no cross-correlation
 * between parameters. In other words the covariance matrix is diagonal.
 */
class GaussianLogPost : public Functor<double>
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
    :Functor<double>(),mMu(0,0),mSigmaInv(0,0)
    {

    }

    /**
     * \brief A constructor that sets up the class
     *
     * \param mu mean of the distribution
     * \param sigmaInv inverse of the digonal covariance matrix, i.e., 1/variance(i)
     */
    GaussianLogPost(realVectorType const& mu,realDiagMatrixType const& sigmaInv)
    :Functor<double>(mu.rows(),1),mMu(mu),mSigmaInv(sigmaInv)
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

    /**
     * \brief an operator that computes the log-posterior probability
     * @return 0 for success
     *
     * This function calls the value() function from the class
     */
    inline int operator()(const realVectorType &x, realVectorType &fvec) const
    {
        value(x,fvec(0));
        return 0;
    }

private:
    realVectorType mMu; /**< the mean of the distribution */
    realDiagMatrixType mSigmaInv; /**< the inverse of the covariance matrix */
};

/**
 * \class logPostNumDiff
 *
 * \brief A class wraps the log-posterior but computes the numerical derivatives
 *
 * \tparam _logPosttype log-posterior type
 *
 * This class wraps the the log-posterior type and uses the value() function
 * to compute numerical derivatives using forward differences.
 */
template<class _logPosttype>
class logPostNumDiff
{
public:

    /**
     * \typedef _logPosttype logPostType
     * \brief log posterior type
     */
    typedef _logPosttype logPostType;

    /**
     * \typedef typename logPostType::realScalarType realScalarType
     * \brief real floating point type
     */
    typedef typename logPostType::realScalarType realScalarType;

    /**
     * \typedef typename logPostType::realVectorType realVectorType
     * \brief real floating point vector type
     */
    typedef typename logPostType::realVectorType realVectorType;

    /**
     * \typedef typename logPostType::indexType indexType
     * \brief integral type
     */
    typedef typename logPostType::indexType indexType;

    /**
     * \typedef Eigen::NumericalDiff<logPostType>  numDiffType
     * \brief numerical differntiation type
     */
    typedef Eigen::NumericalDiff<logPostType>  numDiffType;

    /**
     * \typedef typename numDiffType::JacobianType JacobianType
     * \brief Jacobian matrix type
     */
    typedef typename numDiffType::JacobianType JacobianType;

    /**
     * \brief the default constructor
     * @param lp log-posterior
     */
    logPostNumDiff(logPostType const & lp)
    :mLp(lp),mNumDiff(lp)
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
        return mLp.value(q,val);
    }

    /**
     * \brief compute the derivties of the log-post wrt q
     *
     * \param q the position at which the log-post derivs are sought
     * \param dq the derivatives to be returned
     */
    inline void derivs(realVectorType const & q,realVectorType & dq) const
    {
        JacobianType dqT(1, q.rows());
        mNumDiff.df(q,dqT);
        dq = dqT.transpose();
    }

    /**
     * \brief return the number of dimensions of the posterior distribution
     *
     * \return the number of dimensions of the log-posterior distribution
     */
    inline indexType numDims() const
    {
        return mLp.numDims();
    }

private:
    logPostType const & mLp;
    numDiffType mNumDiff;

};

int main(void)
{
    // define the floating point type
    //typedef double realScalarType;
    // define the potential energy, i.e., the log-posterior
    typedef GaussianLogPost potEngType;
    // real vector for defining log-posterior, derive from the potEngType
    typedef typename potEngType::realVectorType realVectorType;
    // real diagonal-matrix for defining log-posterior , derive from the potEngType
    typedef typename potEngType::realDiagMatrixType realDiagMatrixType;

    // define the log-posterior with numerical differntiation derivs()
    typedef logPostNumDiff<potEngType> logPostNumDiffType;

    // define the sampler using the User Interface
    typedef mpp::canonicalHamiltonianSampler<logPostNumDiffType> samplerUIType;
    // define the seed type
    typedef samplerUIType::seedType seedType;

    // define the diemensionaligy of the problem
    size_t const numParams = 10;

    // make a Gaussian posterior distribution
    realVectorType mu = realVectorType::Zero(numParams);
    realDiagMatrixType sigmaInv = realVectorType::Ones(numParams);
    potEngType G(mu,sigmaInv);

    logPostNumDiffType lPNumDiff(G);

    // define the step size and the number of steps for the integrator
    realScalarType const maxEps = 1;
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

    // define the sampler user interface type
    samplerUIType samplerUI(lPNumDiff,numParams,maxEps,maxNumsteps,startPoint,
        seed,packetSize,numBurn,numSamples,rootPathStr,consoleOutput,
        delimiter,precision,mInv);

    // run the sampler
    samplerUI.run();

    return 0;
}

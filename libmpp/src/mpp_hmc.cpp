#define BOOST_ALL_DYN_LINK
#include <mpp/core>
#include <mpp.h>

template<typename _realScalarType>
class mppLogPost
{
public:
    typedef _realScalarType realScalarType;
    typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realVectorType;
    typedef Eigen::Matrix<realScalarType, Eigen::Dynamic, 1> realDiagMatrixType;
    typedef typename realVectorType::Index indexType;

    mppLogPost(
        void (*p_log_post_func)( realScalarType const*,realScalarType*),
        void (*p_log_post_derivs)(realScalarType const*,realScalarType*),
        indexType const numDims
        )
    {
        m_p_log_post_func = p_log_post_func;
        m_p_log_post_derivs = p_log_post_derivs;
        m_numDims = numDims;
    }

    inline void value(realVectorType  const & q, realScalarType & val) const
    {
        realScalarType const*  p_q = &q(0);
        m_p_log_post_func(p_q,&val);
    }

    inline void derivs(realVectorType  const & q,realVectorType & dq) const
    {
        realScalarType const*  p_q = &q(0);
        realScalarType * p_dq = &dq(0);
        m_p_log_post_derivs(p_q,p_dq);
    }

    inline indexType numDims(void) const
    {
        return m_numDims;
    }

private:
    void (*m_p_log_post_func)( realScalarType const* ,realScalarType*);
    void (*m_p_log_post_derivs)(realScalarType const* ,realScalarType*);
    indexType m_numDims;
};


// define the floating point type and include the dependant code
#define MPP_FLOATING_POINT_TYPE float
#define MPP_FLOATING_POINT_TYPE_FUNC_NAME(x) f_##x
#include "mpp_hmc_impl.cpp"
#undef MPP_FLOATING_POINT_TYPE
#undef MPP_FLOATING_POINT_TYPE_FUNC_NAME

#define MPP_FLOATING_POINT_TYPE double
#define MPP_FLOATING_POINT_TYPE_FUNC_NAME(x) d_##x
#include "mpp_hmc_impl.cpp"
#undef MPP_FLOATING_POINT_TYPE
#undef MPP_FLOATING_POINT_TYPE_FUNC_NAME

#define MPP_FLOATING_POINT_TYPE long double
#define MPP_FLOATING_POINT_TYPE_FUNC_NAME(x) ld_##x
#include "mpp_hmc_impl.cpp"
#undef MPP_FLOATING_POINT_TYPE
#undef MPP_FLOATING_POINT_TYPE_FUNC_NAME

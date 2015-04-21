#ifndef MPP_LEAPFROG_HPP
#define MPP_LEAPFROG_HPP

namespace mpp{ namespace Hamiltonian{

    /**
     * \ingroup Hamiltonian
     *
     * \class leapfrog
     *
     * \brief A class that implemets leapfrog integrator
     *
     * This class implements the leapfrog integrator. MORE INFO TO COME.
     */
    class leapfrog
    {
    public:
        /**
         * \brief Integrate the Hamiltonian
         *
         * \tparam potEngType Potentail Energy type
         * \tparam kinEngType Kinetic Energy type
         *
         * \param q positon vector
         * \param p momentum vector
         * \param eps epsilon or the step size
         * \param numSteps number steps in the integration
         * \param deltaH deifference in Hamiltonian after integration
         *
         * This method integrates the Hamilotian from (p,q) to (p',q') through
         * numSteps steps and epsilon step size.
         */
        template<class potEngType,class kinEngType>
        static void integrate(const typename kinEngType::realScalarType eps,const size_t numSteps,
            potEngType  & G,kinEngType & K,typename kinEngType::realVectorType & q,
            typename kinEngType::realVectorType & p)
        {
            typedef typename potEngType::realVectorType realVectorType;
            typedef typename realVectorType::Index indexType;
            typedef typename potEngType::realScalarType realScalarType;
            typedef typename kinEngType::realScalarType realScalarTypeTypeKE;

            static_assert(std::is_floating_point<realScalarType>::value,
                "PARAMETER SHOULD BE A FLOATING POINT TYPE");
            static_assert(std::is_same<realScalarType,realScalarTypeTypeKE>::value,
                "POTENTIAL ENERGY AND KINTETIC ENERGY SHOULD SHOULD HAVE THE SAME FLOATING POINT TYPE");

            BOOST_ASSERT_MSG(q.rows() == p.rows(),
                "position and momentum should have the same number of dimensions");
            BOOST_ASSERT_MSG(q.rows() == G.numDims(),
                "position and the potentail enegery should have the same number of dimensions");

            BOOST_ASSERT_MSG(K.numDims() == G.numDims(),
                "potentail and kinetic enegeries should have the same number of dimensions");
            BOOST_ASSERT_MSG(eps>0 and eps <2,"For stability of the leapfrog, we require 0<eps<2");

            // check if we need to proceed further
            if(numSteps <= 0)
            {
                return;
            }

            // compute the derivatives
            const indexType N=q.rows();
            realVectorType dp(N);
            realVectorType dq(N);
            G.derivs(q,dq);
            K.derivs(p,dp);

            // take half a step
            p = p + 0.5*eps*dq;

            // now take full steps
            for(size_t i=0;i<numSteps;++i)
            {
                K.derivs(p,dp);
                q = q - eps*dp;
                G.derivs(q,dq);
                p = p + eps*dq;
            }

            // move the momentum back half a step
            p = p - 0.5*eps*dq;
        }
    };

}//namespace Hamiltonian
}//namespace mpp

#endif //MPP_LEAPFROG_HPP

#ifndef MPP_MPP_H
#define MPP_MPP_H

#ifdef __cplusplus
extern "C"
{
#endif

#define MPP_FLOATING_POINT_TYPE float
#define MPP_FLOATING_POINT_TYPE_FUNC_NAME(x) f_##x
#include "mpp_impl.h"
#undef MPP_FLOATING_POINT_TYPE
#undef MPP_FLOATING_POINT_TYPE_FUNC_NAME

#define MPP_FLOATING_POINT_TYPE double
#define MPP_FLOATING_POINT_TYPE_FUNC_NAME(x) d_##x
#include "mpp_impl.h"
#undef MPP_FLOATING_POINT_TYPE
#undef MPP_FLOATING_POINT_TYPE_FUNC_NAME

#define MPP_FLOATING_POINT_TYPE long double
#define MPP_FLOATING_POINT_TYPE_FUNC_NAME(x) ld_##x
#include "mpp_impl.h"
#undef MPP_FLOATING_POINT_TYPE
#undef MPP_FLOATING_POINT_TYPE_FUNC_NAME

// we need to improve this by
// http://stackoverflow.com/questions/147267/easy-way-to-use-variables-of-enum-types-as-string-in-c/202511#202511

#ifdef __cplusplus
}
#endif

#endif /*MPP_MPP_H*/

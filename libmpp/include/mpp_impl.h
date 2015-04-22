void MPP_FLOATING_POINT_TYPE_FUNC_NAME(mpp_hmc_diag_canon)(
    long const* num_params,
    MPP_FLOATING_POINT_TYPE const* max_eps,
    long const* max_num_steps,
    MPP_FLOATING_POINT_TYPE const* start_point,
    long const* rand_seed,
    long const* packet_size,
    long const* num_burn,
    long const* num_samples,
    char const* root_path_str,
    int const* console_output,
    char const* delimiter,
    int const* precision,
    MPP_FLOATING_POINT_TYPE const*  ke_diag_m_inv,
    void (*p_log_post_func)(MPP_FLOATING_POINT_TYPE const* ,MPP_FLOATING_POINT_TYPE*),
    void (*p_log_post_derivs)(MPP_FLOATING_POINT_TYPE const* ,MPP_FLOATING_POINT_TYPE*)
);

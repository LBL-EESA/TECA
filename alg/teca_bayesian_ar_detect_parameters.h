#ifndef teca_bayesian_ar_detect_parameters_h
#define teca_bayesian_ar_detect_parameters_h

#include "teca_algorithm.h"

TECA_SHARED_OBJECT_FORWARD_DECL(teca_bayesian_ar_detect_parameters)

/**
An algorithm that constructs and serves up the parameter
table needed to run the Bayesain AR detector
*/
class teca_bayesian_ar_detect_parameters : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_bayesian_ar_detect_parameters)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_bayesian_ar_detect_parameters)
    TECA_ALGORITHM_CLASS_NAME(teca_bayesian_ar_detect_parameters)
    ~teca_bayesian_ar_detect_parameters();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

protected:
    teca_bayesian_ar_detect_parameters();

private:
    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    struct internals_t;
    internals_t *internals;
};

#endif

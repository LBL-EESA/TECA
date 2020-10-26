#ifndef teca_bayesian_ar_detect_h
#define teca_bayesian_ar_detect_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_bayesian_ar_detect)

/// CASCADE BARD atmospheric river detector
/**
Given a point wise IVT (integrated vapor transport) field and a training
parameter table computes the point wise probability of an atmospheric river
using the CASCADE BARD algorithm.

Required inputs:

    1. IVT (integrated vapor transport) array on a Cartesian nesh.
    2. a compatible parameter table. columns of which are : min IVT,
       component area, HWHM lattitude

The names of the input varibale and columns can be specified at run time
through algorithm properties.

Produces:

    A Cartesian mesh with probability of an AR stored in the point centered
    array named "ar_probability". The diagnostic quantites "ar_count" amd
    "parameter_table_row" are stored in information arrays.

For more information see:

Detection of Atmospheric Rivers with Inline Uncertainty Quantification: TECA-BARD v1.0
O'Brien, T. A et al. Geoscientific Model Development, 2020
*/
class teca_bayesian_ar_detect : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_bayesian_ar_detect)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_bayesian_ar_detect)
    TECA_ALGORITHM_CLASS_NAME(teca_bayesian_ar_detect)
    ~teca_bayesian_ar_detect();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the name of the input array
    TECA_ALGORITHM_PROPERTY(std::string, ivt_variable)

    // set the names of columns in the parameter table.
    TECA_ALGORITHM_PROPERTY(std::string, min_ivt_variable)
    TECA_ALGORITHM_PROPERTY(std::string, min_component_area_variable)
    TECA_ALGORITHM_PROPERTY(std::string, hwhm_latitude_variable)

    // flag indicating verbose terminal output is desired.
    // default is 0
    TECA_ALGORITHM_PROPERTY(int, verbose)

    // set/get the number of threads in the pool. setting
    // to -1 results in a thread per core factoring in all MPI
    // ranks running on the node. the default is -1.
    void set_thread_pool_size(int n_threads);
    unsigned int get_thread_pool_size() const noexcept;

    // override the input connections because we are going to
    // take the first input and use it to generate metadata.
    // the second input then becomes the only one the pipeline
    // knows about.
    void set_input_connection(unsigned int id,
        const teca_algorithm_output_port &port) override;

protected:
    teca_bayesian_ar_detect();

    std::string get_label_variable(const teca_metadata &request);

private:
    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    void set_modified() override;

private:
    std::string ivt_variable;
    std::string min_component_area_variable;
    std::string min_ivt_variable;
    std::string hwhm_latitude_variable;
    int thread_pool_size;
    int verbose;

    struct internals_t;
    internals_t *internals;
};

#endif

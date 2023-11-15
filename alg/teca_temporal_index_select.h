#ifndef teca_temporal_index_select_h
#define teca_temporal_index_select_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_temporal_index_select)

/// An algorithm that selects specific time indices
class TECA_EXPORT teca_temporal_index_select : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_temporal_index_select)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_temporal_index_select)
    TECA_ALGORITHM_CLASS_NAME(teca_temporal_index_select)
    ~teca_temporal_index_select();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name indices
     * Set the time axis inidices to select
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(long long, indice)
    ///@}

protected:
    teca_temporal_index_select();

private:
    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::vector<long long> indices;
};

#endif

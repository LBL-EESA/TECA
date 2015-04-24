#ifndef teca_table_reduce_h
#define teca_table_reduce_h

#include "teca_shared_object.h"
#include "teca_dataset_fwd.h"
#include "teca_metadata.h"
#include "teca_temporal_reduction.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_table_reduce)

// a reduction on tabular data over time steps
/**
a reduction on tabular data over time steps.
tabular data from each time step is collected and
concatenated into a big table.
*/
class teca_table_reduce : public teca_temporal_reduction
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_table_reduce)
    ~teca_table_reduce(){}

protected:
    teca_table_reduce();

    // overrides
    p_teca_dataset reduce(
        const const_p_teca_dataset &left,
        const const_p_teca_dataset &right) override;

    std::vector<teca_metadata> initialize_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    teca_metadata initialize_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;
};

#endif

#ifndef teca_table_reduce_h
#define teca_table_reduce_h

#include "teca_shared_object.h"
#include "teca_dataset.h"
#include "teca_metadata.h"
#include "teca_index_reduce.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_table_reduce)

/// A reduction on tabular data over time steps.
/**
 * Tabular data from each time step is collected and
 * concatenated into a big table.
 */
class teca_table_reduce : public teca_index_reduce
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_table_reduce)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_table_reduce)
    TECA_ALGORITHM_CLASS_NAME(teca_table_reduce)
    ~teca_table_reduce(){}

protected:
    teca_table_reduce();

    // overrides
    p_teca_dataset reduce(const const_p_teca_dataset &left,
        const const_p_teca_dataset &right) override;

    std::vector<teca_metadata> initialize_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    teca_metadata initialize_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;
};

#endif

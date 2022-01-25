#ifndef teca_cf_time_axis_data_reduce_h
#define teca_cf_time_axis_data_reduce_h

#include "teca_config.h"
#include "teca_shared_object.h"
#include "teca_dataset.h"
#include "teca_metadata.h"
#include "teca_index_reduce.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_cf_time_axis_data_reduce)

/** @brief
 * Gathers the time axis and metadata from a parallel read of a
 * set of NetCDF CF2 files.
 */
class TECA_EXPORT teca_cf_time_axis_data_reduce : public teca_index_reduce
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_cf_time_axis_data_reduce)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_cf_time_axis_data_reduce)
    TECA_ALGORITHM_CLASS_NAME(teca_cf_time_axis_data_reduce)
    ~teca_cf_time_axis_data_reduce() override = default;

protected:
    teca_cf_time_axis_data_reduce();

    // overrides
    p_teca_dataset reduce(int device_id, const const_p_teca_dataset &left,
        const const_p_teca_dataset &right) override;

    std::vector<teca_metadata> initialize_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    teca_metadata initialize_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;
};

#endif

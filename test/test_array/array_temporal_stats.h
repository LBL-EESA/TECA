#ifndef array_temporal_stats_h
#define array_temporal_stats_h

#include "teca_config.h"
#include "teca_shared_object.h"
#include "array.h"

#include "teca_index_reduce.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(array_temporal_stats)

/** example demonstarting a temporal reduction. min, average
 and max are computed over time steps for the named array.
*/
class TECA_EXPORT array_temporal_stats : public teca_index_reduce
{
public:
    TECA_ALGORITHM_STATIC_NEW(array_temporal_stats)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(array_temporal_stats)
    TECA_ALGORITHM_CLASS_NAME(array_temporal_stats)
    ~array_temporal_stats(){}

    // set the array to process
    TECA_ALGORITHM_PROPERTY(std::string, array_name)

private:
    // helpers
    p_array new_stats_array();
    p_array new_stats_array(const_p_array input);
    p_array new_stats_array(const_p_array l_input, const_p_array r_input);

protected:
    array_temporal_stats();

    // overrides
    p_teca_dataset reduce(int device_id, const const_p_teca_dataset &left,
        const const_p_teca_dataset &right) override;

    p_teca_dataset finalize(int device_id,
        const const_p_teca_dataset &ds) override;

    std::vector<teca_metadata> initialize_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    teca_metadata initialize_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

private:
    std::string array_name;
};

#endif

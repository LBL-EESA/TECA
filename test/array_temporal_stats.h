#ifndef array_temporal_stats_h
#define array_temporal_stats_h

#include "array_fwd.h"

#include "teca_temporal_reduction.h"
#include "teca_meta_data.h"

#include <memory>
#include <string>
#include <vector>

class array_temporal_stats;
typedef std::shared_ptr<array_temporal_stats> p_array_temporal_stats;


/** example demonstarting a temporal reduction. min, average
 and max are computed over time steps for the named array.
*/
class array_temporal_stats : public teca_temporal_reduction
{
public:
    TECA_ALGORITHM_STATIC_NEW(array_temporal_stats)
    ~array_temporal_stats(){}

    // set the array to process
    TECA_ALGORITHM_PROPERTY(std::string, array_name)

private:
    // helpers
    p_array new_stats_array();
    p_array new_stats_array(p_array input);
    p_array new_stats_array(p_array l_input, p_array r_input);

protected:
    array_temporal_stats();

    // overrides
    p_teca_dataset reduce(
        const p_teca_dataset &left,
        const p_teca_dataset &right) override;

    std::vector<teca_meta_data> initialize_upstream_request(
        unsigned int port,
        const std::vector<teca_meta_data> &input_md,
        const teca_meta_data &request) override;

    teca_meta_data initialize_output_meta_data(
        unsigned int port,
        const std::vector<teca_meta_data> &input_md) override;

private:
    std::string array_name;
};

#endif

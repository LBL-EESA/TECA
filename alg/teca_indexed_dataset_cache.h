#ifndef teca_indexed_dataset_cache_h
#define teca_indexed_dataset_cache_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_indexed_dataset_cache)

/// Caches N datasets such that repeated requests for the same dataset are served from the cache
/**
 * A cache storing up to N datasets. Datasets are identified using their
 * request index.  Repeated requests for the same dataset (ie same index) are
 * served from the cache. When more than N unique datasets have been requested
 * the cache is modified such that the least recently used dataset is replaced.
 */
class TECA_EXPORT teca_indexed_dataset_cache : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_indexed_dataset_cache)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_indexed_dataset_cache)
    TECA_ALGORITHM_CLASS_NAME(teca_indexed_dataset_cache)
    ~teca_indexed_dataset_cache();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name max_cache_size
     * Set the max number of datasets to cache.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(unsigned long, max_cache_size)
    ///@}

    /// clear any cached data.
    void clear_cache();

    /** @name override_request_index
     * When set the request is modified to always request index 0. This can be
     * used to cache data that doesn't change.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, override_request_index)
    ///@}

protected:
    teca_indexed_dataset_cache();

private:

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    unsigned long max_cache_size;
    int override_request_index;

    struct internals_t;
    internals_t *internals;
};

#endif

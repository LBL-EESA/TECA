#ifndef teca_dataset_source_h
#define teca_dataset_source_h

#include "teca_config.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_dataset.h"
#include "teca_shared_object.h"

#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_dataset_source)

/// An algorithm that serves up user provided data and metadata.
/**
 * This algorithm can be used to inject a dataset constructed
 * on outside of TECA into a TECA pipeline.
 */
class TECA_EXPORT teca_dataset_source : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_dataset_source)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_dataset_source)
    TECA_ALGORITHM_CLASS_NAME(teca_dataset_source)
    ~teca_dataset_source();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set/get the dataset to insert into the pipeline
    TECA_ALGORITHM_VECTOR_PROPERTY(p_teca_dataset, dataset)

    // set/get the metadata to insert into the pipeline
    TECA_ALGORITHM_PROPERTY(teca_metadata, metadata)

protected:
    teca_dataset_source();

private:
    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::vector<p_teca_dataset> datasets;
    teca_metadata metadata;
};

#endif

#ifndef teca_dataset_source_h
#define teca_dataset_source_h

#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_dataset_fwd.h"
#include "teca_shared_object.h"

TECA_SHARED_OBJECT_FORWARD_DECL(teca_dataset_source)

/**
An algorithm that serves up user provided data and metadata.
This algorithm can be used to inject a dataset constructed
on outside of TECA into a TECA pipleine.
*/
class teca_dataset_source : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_dataset_source)
    ~teca_dataset_source();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set/get the dataset to insert into the pipeline
    TECA_ALGORITHM_PROPERTY(p_teca_dataset, dataset)

    // set/get the metadata to insert into the pipeline
    TECA_ALGORITHM_PROPERTY(teca_metadata, metadata)

protected:
    teca_dataset_source();

private:
    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    p_teca_dataset dataset;
    teca_metadata metadata;
};

#endif

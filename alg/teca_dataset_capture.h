#ifndef teca_dataset_capture_h
#define teca_dataset_capture_h

#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_dataset_fwd.h"
#include "teca_shared_object.h"

TECA_SHARED_OBJECT_FORWARD_DECL(teca_dataset_capture)

/**
An algorithm that takes a reference to dataset produced
by the upstream algorithm it is connected to. The dataset
is passed through so that this can be inserted anywhere
giving one access to the intermediate data.
*/
class teca_dataset_capture : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_dataset_capture)
    ~teca_dataset_capture();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set/get the dataset from the last pipeline update
    TECA_ALGORITHM_PROPERTY(const_p_teca_dataset, dataset)

protected:
    teca_dataset_capture();

private:
    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    const_p_teca_dataset dataset;
};

#endif

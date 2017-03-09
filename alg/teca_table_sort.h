#ifndef teca_table_sort_h
#define teca_table_sort_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_table_sort)

/// an algorithm that sorts a table in ascending order
class teca_table_sort : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_table_sort)
    ~teca_table_sort();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the column to sort by
    TECA_ALGORITHM_PROPERTY(std::string, index_column)
    TECA_ALGORITHM_PROPERTY(int, index_column_id)

    // enable/disable stable sorting. default 0
    TECA_ALGORITHM_PROPERTY(int, stable_sort)

    void enable_stable_sort(){ set_stable_sort(1); }
    void disable_stable_sort(){ set_stable_sort(0); }

protected:
    teca_table_sort();

private:
    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string index_column;
    int index_column_id;
    int stable_sort;
};

#endif

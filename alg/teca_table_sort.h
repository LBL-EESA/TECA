#ifndef teca_table_sort_h
#define teca_table_sort_h

#include "teca_config.h"
#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_table_sort)

/** @breif An algorithm that sorts the rows of a table using the values in a
 * specified column.
 * @details The sort can be done in ascending or descending order and a stable
 * algorithm (i.e. one which does not change the order of equivalent elements)
 * can be used. The table sort is especially useful when TECA's threaded
 * algorithms, such as ::teca_table_reduce, produce in tables with a
 * non-deterministic order.
 */
class TECA_EXPORT teca_table_sort : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_table_sort)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_table_sort)
    TECA_ALGORITHM_CLASS_NAME(teca_table_sort)
    ~teca_table_sort();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name index_column
     * Set the name of the column to sort by.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, index_column)
    ///@}

    /** @name index_column_id
     * Set the number of the column to sort by.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, index_column_id)
    ///@}

    /** @name stable_sort
     * Enable or disable stable sort. The default is disabled.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, stable_sort)

    /// Enables stable sort.
    void enable_stable_sort(){ set_stable_sort(1); }

    /// Disable stable sort
    void disable_stable_sort(){ set_stable_sort(0); }
    ///@}

    /** @name ascending_order
     * Set the sort order to ascending. the default is descending.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, ascending_order)
    ///@}


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
    int ascending_order;
};

#endif

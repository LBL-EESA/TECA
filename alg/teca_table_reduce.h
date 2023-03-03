#ifndef teca_table_reduce_h
#define teca_table_reduce_h

#include "teca_config.h"
#include "teca_shared_object.h"
#include "teca_dataset.h"
#include "teca_metadata.h"
#include "teca_index_reduce.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_table_reduce)

/// A reduction on tabular data over time steps.
/** Tabular data from the inputs is concatenated by row into a single table.
 * Threading results in out of order execution and generation of the inputs. It
 * may be necessary to sort the data to regain order of the requests. See
 * teca_table_sort.
 */
class TECA_EXPORT teca_table_reduce : public teca_index_reduce
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_table_reduce)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_table_reduce)
    TECA_ALGORITHM_CLASS_NAME(teca_table_reduce)
    ~teca_table_reduce(){}

protected:
    teca_table_reduce();

    // overrides
    p_teca_dataset reduce(int device_id, const const_p_teca_dataset &left,
        const const_p_teca_dataset &right) override;
};

#endif

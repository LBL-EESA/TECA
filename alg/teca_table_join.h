#ifndef teca_table_join_h
#define teca_table_join_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_table_join)

/** @brief An algorithm that joins the columns of tables generated on
 * each of its inputs via a union.
 * @details One must explicitly set the number of inputs and connect an
 * upstream pipeline to each. Metadata including array names and attributes
 * from all inputs are merged into the report. Incoming requests are forwarded
 * to all inputs with the specific arrays belonging to the other inputs
 * removed.  When two inputs provide the same array, the data is fetched from
 * the lower input.
 */
class TECA_EXPORT teca_table_join : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_table_join)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_table_join)
    TECA_ALGORITHM_CLASS_NAME(teca_table_join)
    ~teca_table_join();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** Set the number input connections. The default is 2, this must be called
     * when more than 2 tables are to be merged.
     */
    void set_number_of_input_connections(unsigned int n)
    { this->teca_algorithm::set_number_of_input_connections(n); }

protected:
    teca_table_join();

private:
    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
};

#endif
